from typing import Literal, Sequence, Self
from pathlib import Path

import torch
import torch.nn as nn
import lightning.pytorch as pl

from delpi.model.mae_encoder import Encoder
from delpi.model.input import TheoPeakInput, ExpPeakInput


class DelPiModel(pl.LightningModule):
    """Inference model: encoder + optional split_proj + classifier head.

    Also serves as the base class for DelPiClassifier (fine-tuning).
    Use :meth:`load` to restore a model exported by
    ``DelPiClassifier.export()``::

        model = DelPiModel.load("path/to/model.pth")
    """

    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 12,
        drop_path_rate: float = 0.0,
        global_pool: Literal["token", "avg", "split"] = "token",
        layers: Sequence[int] = [64, 32],
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        if global_pool not in ("token", "avg", "split"):
            raise ValueError(
                f"global_pool must be 'token', 'avg', or 'split', got {global_pool!r}"
            )
        self.global_pool = global_pool
        self.transform = None

        self.encoder = Encoder(
            theo_peak_dim=len(TheoPeakInput),
            exp_peak_dim=len(ExpPeakInput),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            qkv_bias=True,
            drop_path_rate=drop_path_rate,
        )

        # "split": project [CLS; mean(theo); mean(exp)] → embed_dim.
        # concat allows the projection to learn cross-stream interactions
        # (CLS↔theo, CLS↔exp, theo↔exp), which is essential for evidence matching.
        # LayerNorm is intentionally omitted here; the classifier head's leading
        # LayerNorm normalizes uniformly across all pool modes.
        if global_pool == "split":
            self.split_proj = nn.Linear(embed_dim * 3, embed_dim, bias=False)
        else:
            self.split_proj = None

        fc_layers = [nn.LayerNorm(embed_dim, eps=1e-6)]
        num_neurons = [embed_dim] + list(layers)
        for i in range(len(num_neurons) - 1):
            fc_layers.append(nn.Linear(num_neurons[i], num_neurons[i + 1]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
        fc_layers.append(nn.Linear(num_neurons[-1], 1))
        self.classifier = nn.Sequential(*fc_layers)

    def _pool_features(self, x: torch.Tensor, n_theo_tokens: int) -> torch.Tensor:
        """Pool encoder output according to self.global_pool.

        Token layout after encoder: [CLS | theo_tokens (n_theo_tokens) | exp_tokens]

        Args:
            x: encoder output of shape [B, 1 + n_theo + n_exp, embed_dim]
            n_theo_tokens: number of theoretical peak tokens
        Returns:
            Pooled feature tensor of shape [B, embed_dim]
        """
        if self.global_pool == "token":
            return x[:, 0]
        elif self.global_pool == "avg":
            return x[:, 1:].mean(dim=1)
        else:  # "split"
            cls = x[:, 0]
            theo_avg = x[:, 1 : 1 + n_theo_tokens].mean(dim=1)
            exp_avg = x[:, 1 + n_theo_tokens :].mean(dim=1)
            return self.split_proj(torch.cat([cls, theo_avg, exp_avg], dim=-1))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a DelPiModel from a file exported by ``DelPiClassifier.export``.

        Args:
            path: Path to the exported ``.pth`` file containing
                  ``model_state_dict`` and ``hyper_parameters``.

        Returns:
            A :class:`DelPiModel` instance with loaded weights (eval mode, CPU).
        """

        payload = torch.load(Path(path), weights_only=True)
        model = cls(**payload["hyper_parameters"])
        model.load_state_dict(payload["model_state_dict"])
        model.meta = payload.get("meta", {})
        model.eval()
        return model

    def forward(
        self,
        x_theo: torch.Tensor,
        x_exp: torch.Tensor,
        return_feature: bool = False,
    ):
        if self.transform is not None:
            x_theo, x_exp = self.transform(x_theo, x_exp)

        x = self.encoder._forward_impl(x_theo, x_exp)
        x_feature = self._pool_features(x, n_theo_tokens=x_theo.shape[1])
        logits = self.classifier(x_feature)

        if return_feature:
            return logits, x_feature
        return logits
