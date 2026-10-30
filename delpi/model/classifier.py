from typing import Literal, Sequence, Self
from pathlib import Path

import torch
import torch.nn as nn
import lightning.pytorch as pl

from delpi.model.mae_encoder import Encoder
from delpi.model.input import INPUT_SCHEMA_VERSION, TheoPeakInput, ExpPeakInput
from delpi.model.feature_encoder import valid_tokens


class DelPiModel(pl.LightningModule):
    """Inference model: encoder + optional split_proj + classifier head.

    Also serves as the base class for DelPiClassifier (fine-tuning).
    Use :meth:`load` to restore a model exported by
    ``DelPiClassifier.export()``::

        model = DelPiModel.load("path/to/model.pth")
    """

    INPUT_SCHEMA_VERSION = INPUT_SCHEMA_VERSION

    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 12,
        drop_path_rate: float = 0.0,
        global_pool: Literal["token", "avg", "split"] = "token",
        layers: Sequence[int] = [64, 32],
        dropout: float = 0.1,
        input_schema_version: int = INPUT_SCHEMA_VERSION,
        **kwargs,
    ):
        super().__init__()

        if global_pool not in ("token", "avg", "split"):
            raise ValueError(
                f"global_pool must be 'token', 'avg', or 'split', got {global_pool!r}"
            )
        if input_schema_version != self.INPUT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported DelPi input schema; retrain the model for schema {INPUT_SCHEMA_VERSION}"
            )
        self.save_hyperparameters(ignore=["kwargs"])
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

    def _pool_features(self, x, n_theo_tokens, theo_valid_mask, exp_valid_mask):
        """Pool [CLS, context, theoretical, experimental], excluding padding."""

        def mean(tokens, valid):
            weights = valid.unsqueeze(-1).to(tokens.dtype)
            return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1)

        if self.global_pool == "token":
            return x[:, 0]
        if self.global_pool == "avg":
            valid = torch.cat(
                (
                    torch.ones((x.shape[0], 1), dtype=torch.bool, device=x.device),
                    theo_valid_mask,
                    exp_valid_mask,
                ),
                dim=1,
            )
            return mean(x[:, 1:], valid)
        theo_avg = mean(x[:, 2 : 2 + n_theo_tokens], theo_valid_mask)
        exp_avg = mean(x[:, 2 + n_theo_tokens :], exp_valid_mask)
        return self.split_proj(torch.cat((x[:, 0], theo_avg, exp_avg), dim=-1))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a DelPiModel from a file exported by ``DelPiClassifier.export``."""
        from delpi.utils.model_io import load_model

        return load_model(cls, path)

    def forward(
        self,
        x_theo: torch.Tensor,
        x_exp: torch.Tensor,
        x_context: torch.Tensor,
        theo_valid_mask: torch.Tensor,
        exp_valid_mask: torch.Tensor | None = None,
        return_feature: bool = False,
    ):
        if self.transform is not None:
            x_theo, x_exp, x_context = self.transform(x_theo, x_exp, x_context)

        theo_valid_mask = valid_tokens(x_theo, theo_valid_mask)
        exp_valid_mask = valid_tokens(x_exp, exp_valid_mask)
        x = self.encoder._forward_impl(
            x_theo, x_exp, x_context, theo_valid_mask, exp_valid_mask
        )
        x_feature = self._pool_features(
            x, x_theo.shape[1], theo_valid_mask, exp_valid_mask
        )
        logits = self.classifier(x_feature)

        if return_feature:
            return logits, x_feature
        return logits
