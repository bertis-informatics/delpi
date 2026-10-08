from typing import Dict, Self, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
import torchmetrics
from torch.utils.data import DataLoader, Dataset

from delpi.constants import COMMON_NEUTRAL_LOSS_INDEX
from delpi.lcms.neutral_loss import NeutralLoss
from delpi.model.spec_lib.block import ResNet1D, BiLSTM, Transformer, Permute
from delpi.model.pos_encoder import PositionalEncoding
from delpi.utils.metric import SpectralAngle
from delpi.search.tl.lr_decay import param_groups_lrd
from delpi.utils.scheduler import get_cosine_schedule_with_warmup
from delpi.model.spec_lib.aa_encoder import MOD_FEATURE_MAP

EPS = 1e-9


class Ms2SpectrumPredictor(LightningModule):
    """Predict four b/y charge channels per neutral loss, in the supplied order.

    The default preserves legacy 8-channel checkpoints and search callers.
    Pass COMMON_NEUTRAL_LOSSES when training the 16-channel Prospect model.
    """

    def __init__(
        self,
        aa_vocab_size: int = 22,
        aa_embedding_dim: int = 32,
        mod_embedding_dim: int = 8,
        meta_embedding_dim: int = 4,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 1,
        max_lr: float = 1e-3,
        num_warmup_steps: int = 4,
        num_training_steps: int = 40,
        encoder_type: str = "transformer",
        # Transformer specific parameters
        transformer_depth: int = 4,
        transformer_num_heads: int = 8,
        transformer_qkv_bias: bool = True,
        transformer_drop_path_rate: float = 0.0,
        layer_decay: float = 1.0,
        neutral_losses: Sequence[str] = (
            NeutralLoss.NO_LOSS.symbol,
            NeutralLoss.H3O4P.symbol,
        ),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        neutral_losses = tuple(neutral_losses)
        if (
            not neutral_losses
            or neutral_losses[0] != NeutralLoss.NO_LOSS.symbol
            or len(set(neutral_losses)) != len(neutral_losses)
            or any(loss not in COMMON_NEUTRAL_LOSS_INDEX for loss in neutral_losses)
        ):
            raise ValueError(
                "neutral_losses must contain unique DelPi neutral-loss symbols "
                f"with {NeutralLoss.NO_LOSS.symbol!r} first."
            )
        self.neutral_losses = neutral_losses
        self.num_fragment_channels = 4 * len(neutral_losses)
        self.fragment_types = [
            f"{ion_type}{NeutralLoss.get(loss).name}_z{charge}"
            for loss in neutral_losses
            for ion_type in ("b", "y")
            for charge in (1, 2)
        ]
        self.encoder_type = encoder_type

        # Embedding layer
        self.aa_embedding = nn.Embedding(
            aa_vocab_size, aa_embedding_dim - mod_embedding_dim - meta_embedding_dim
        )
        self.mod_embedding = nn.Linear(MOD_FEATURE_MAP.shape[-1], mod_embedding_dim)
        self.meta_embedding = nn.Linear(4, meta_embedding_dim)

        # Choose encoder architecture
        if encoder_type == "cnn_rnn":
            # CNN + RNN encoder
            self.encoder = nn.Sequential(
                Permute(0, 2, 1),  # [B, L, D] -> [B, D, L] for CNN
                ResNet1D(
                    in_channels=aa_embedding_dim,
                    out_channels=embedding_dim,
                    conv1_kernel_size=5,
                ),
                Permute(0, 2, 1),  # [B, D, L] -> [B, L, D] for RNN
                PositionalEncoding(embedding_dim),
                BiLSTM(
                    embedding_dim=embedding_dim,
                    num_layers=num_layers,
                    return_sequences=True,  # Return all sequence outputs for MS2
                ),
            )
            encoder_output_dim = 2 * embedding_dim  # Bidirectional LSTM

        elif encoder_type == "transformer":
            # Transformer encoder
            self.encoder = nn.Sequential(
                nn.Linear(aa_embedding_dim, embedding_dim),
                PositionalEncoding(embedding_dim),
                Transformer(
                    embed_dim=embedding_dim,
                    depth=transformer_depth,
                    num_heads=transformer_num_heads,
                    qkv_bias=transformer_qkv_bias,
                    drop_path_rate=transformer_drop_path_rate,
                    return_sequences=True,  # Return all sequence outputs for MS2
                ),
            )
            encoder_output_dim = embedding_dim

        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. Choose 'cnn_rnn' or 'transformer'"
            )

        # Output layer for fragment intensities
        # Four channels [b_z1, b_z2, y_z1, y_z2] per neutral loss.
        # Input: (B, L-1, encoder_output_dim) -> (B, L-1, num_fragment_channels)
        self.fragment_predictor = nn.Sequential(
            nn.Linear(encoder_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.num_fragment_channels),
            nn.ReLU(),  # Ensure non-negative intensities
        )

        # Metrics
        self.train_corr = torchmetrics.PearsonCorrCoef()
        self.valid_corr = nn.ModuleDict(
            {loss: torchmetrics.PearsonCorrCoef() for loss in self.neutral_losses}
        )
        self.train_sa = SpectralAngle()
        self.valid_sa = nn.ModuleDict(
            {loss: SpectralAngle() for loss in self.neutral_losses}
        )

        self.max_lr = max_lr
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.layer_decay = layer_decay

        self.save_hyperparameters()

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load an Ms2SpectrumPredictor from a file exported by :meth:`export`."""
        from delpi.utils.model_io import load_model

        return load_model(cls, path)

    def export(
        self,
        save_path: str | Path,
        model_version: str = "v1.0",
    ) -> None:
        """Export weights + hyperparameters + meta to ``save_path``.

        The exported file can be reconstructed without Lightning::

            model = Ms2SpectrumPredictor.load(save_path)
        """
        from delpi.utils.model_io import save_model

        save_model(self, save_path, model_version=model_version)

    def forward(self, x_aa, x_mod, x_meta):
        """
        Forward pass for MS2 spectrum prediction.

        Args:
            x_aa: Amino acid sequence tensor (B, L+2) where L is peptide length
            x_mod: Encoded modification features (B, L+2, number of elements)
            x_meta: Precursor charge, NCE, fragmentation, mass analyzer (B, 4)

        Returns:
            Fragment intensities (B, L-1, 4 * len(neutral_losses)), with
            [b_z1, b_z2, y_z1, y_z2] in each neutral-loss block.
        """
        # Pass through AA and Mod embedding layers
        x_aa_emb = self.aa_embedding(x_aa.to(torch.int32))
        x_mod_emb = self.mod_embedding(x_mod)
        x_meta_emb = self.meta_embedding(x_meta)[:, None, :].expand(
            -1, x_aa_emb.size(1), -1
        )
        x_emb = torch.cat([x_aa_emb, x_mod_emb, x_meta_emb], dim=-1)

        # Pass through encoder (both CNN+RNN and Transformer are now Sequential)
        x_emb = self.encoder(x_emb)

        # Select one position per cleavage: [B, L+2, D] -> [B, L-1, D].
        x_emb = x_emb[:, 2:-1, :]

        # Predict fragment intensities
        y_pred = self.fragment_predictor(x_emb)

        return y_pred

    @torch.inference_mode()
    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> pl.DataFrame:
        """Return normalized intensities for each precursor and cleavage.

        Columns follow the configured neutral-loss order and are named
        "{ion_type}{neutral_loss.name}_z{charge}", e.g. "b_z1", "y-H2O_z1",
        or "y-H3O4P_z2". All configured losses are returned.
        Each spectrum is normalized by the maximum of its returned channels.
        """

        precursor_index_arr = batch["precursor_index"].to(
            device="cpu", dtype=torch.uint32
        ).numpy()
        x_aa = batch["x_aa"].to(device=self.device)
        x_mod = batch["x_mod"].to(device=self.device)
        x_meta = batch["x_meta"].to(device=self.device)
        cleavage_count = x_aa.shape[-1] - 3

        y_pred = self(x_aa, x_mod, x_meta)
        scale = torch.amax(y_pred, dim=(1, 2), keepdim=True)
        y_pred = y_pred / (scale + EPS)
        y_pred = y_pred.detach().cpu().numpy()
        ion_type_count = y_pred.shape[-1]

        batch_ms2_df = pl.from_numpy(
            y_pred.reshape(-1, ion_type_count),
            schema=self.fragment_types,
            orient="row",
        ).select(
            pl.Series(
                name="precursor_index",
                values=precursor_index_arr.repeat(cleavage_count),
                dtype=pl.UInt32,
            ),
            pl.Series(
                name="cleavage_index",
                values=np.tile(
                    np.arange(cleavage_count, dtype=np.uint8), y_pred.shape[0]
                ),
                dtype=pl.UInt8,
            ),
            pl.col(*self.fragment_types),
        )

        return batch_ms2_df

    def _compute_loss(self, x_aa, x_mod, x_meta, y_true):
        """
        Compute loss for MS2 spectrum prediction.

        Args:
            x_aa: Amino acid sequence tensor
            x_mod: Modification tensor
            y_true: True fragment intensities (B, L-1, num_fragment_channels)

        Returns:
            loss, y_true, y_pred
        """
        y_pred = self(x_aa, x_mod, x_meta)
        # Legacy search fine-tuning can supply only the four no-loss channels.
        if self.neutral_losses == (
            NeutralLoss.NO_LOSS.symbol,
            NeutralLoss.H3O4P.symbol,
        ) and y_true.size(-1) == 4:
            y_pred = y_pred[..., :4]
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"MS2 target shape {tuple(y_true.shape)} does not match model "
                f"output {tuple(y_pred.shape)} for neutral_losses={self.neutral_losses}. "
                "Use targets with the same neutral-loss channel order as the model."
            )

        # Supervise every fragment channel, including all configured neutral losses.
        loss = nn.functional.mse_loss(y_pred, y_true)

        return loss, y_true, y_pred

    def training_step(self, batch, batch_idx):
        """Training step for MS2 spectrum prediction."""

        x_aa = batch["x_aa"]
        x_mod = batch["x_mod"]
        x_meta = batch["x_meta"]
        y_true = batch["y_intensity"]  # Expected key for MS2 data
        batch_size = len(y_true)

        loss, y_true, y_pred = self._compute_loss(x_aa, x_mod, x_meta, y_true)

        # Flatten for correlation calculation
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        self.train_corr.update(y_pred_flat, y_true_flat)
        self.train_sa.update(y_pred_flat, y_true_flat)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train_corr",
            self.train_corr,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_sa",
            self.train_sa,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for MS2 spectrum prediction."""
        x_aa = batch["x_aa"]
        x_mod = batch["x_mod"]
        x_meta = batch["x_meta"]
        y_true = batch["y_intensity"]
        batch_size = len(y_true)

        loss, y_true, y_pred = self._compute_loss(x_aa, x_mod, x_meta, y_true)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        for loss_index, neutral_loss in enumerate(
            self.neutral_losses[: y_true.size(-1) // 4]
        ):
            channels = slice(4 * loss_index, 4 * (loss_index + 1))
            loss_pred = y_pred[..., channels]
            loss_true = y_true[..., channels]
            # Pearson accumulates all values of this loss across validation.
            corr = self.valid_corr[neutral_loss]
            corr.update(loss_pred.reshape(-1), loss_true.reshape(-1))
            # Spectral angle is computed per spectrum before averaging.
            sa = self.valid_sa[neutral_loss]
            sa.update(
                loss_pred.reshape(batch_size, -1),
                loss_true.reshape(batch_size, -1),
            )
            self.log(
                f"val_loss_{neutral_loss}",
                nn.functional.mse_loss(loss_pred, loss_true),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                f"val_corr_{neutral_loss}",
                corr,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"val_sa_{neutral_loss}",
                sa,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        return loss

    def configure_optimizers(self):
        """Configure AdamW with epoch-based warmup and cosine decay.

        num_warmup_steps and num_training_steps count epochs for this model.
        """
        if self.layer_decay < 1.0:
            param_groups = param_groups_lrd(
                model=self,
                weight_decay=0.05,
                layer_decay=self.layer_decay,
                max_lr=self.max_lr,
            )
        else:
            param_groups = self.parameters()

        optimizer = torch.optim.AdamW(param_groups, lr=self.max_lr, weight_decay=0.05)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            min_lr=1e-6,
        )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            },
        )

    def get_trainset(self):
        if self.fractions is not None:
            n = len(self.original_train_ds)
            subset_size = max(1, int(n * self.fractions))
            max_non_overlap_epochs = max(1, n // subset_size)

            # Consume non-overlapping chunks from one global permutation,
            # then reshuffle per cycle for better dataset coverage.
            cycle_epochs = self.subset_shuffle_every_epochs
            if cycle_epochs is None:
                cycle_epochs = max_non_overlap_epochs
            cycle_epochs = max(1, min(cycle_epochs, max_non_overlap_epochs))

            cycle_index = self.current_epoch // cycle_epochs
            epoch_in_cycle = self.current_epoch % cycle_epochs

            if (
                self._subset_perm is None
                or self._subset_perm_cycle_index != cycle_index
                or self._subset_perm_size != n
            ):
                rng = np.random.default_rng(self.subset_seed + cycle_index)
                self._subset_perm = rng.permutation(n)
                self._subset_perm_cycle_index = cycle_index
                self._subset_perm_size = n

            start = epoch_in_cycle * subset_size
            stop = min(start + subset_size, n)
            subset_idx = self._subset_perm[start:stop]

            return self.original_train_ds.make_subset_from_indices(subset_idx)

        return self.original_train_ds

    def set_dataset(
        self,
        train_dataset,
        val_dataset,
        batch_size,
        fractions=None,
        num_workers=8,
        subset_shuffle_every_epochs=None,
        subset_seed=0,
    ):
        """Set datasets for training and validation."""
        self.original_train_ds = train_dataset
        self.val_ds = val_dataset
        self.batch_size = batch_size
        self.fractions = fractions
        self.num_workers = num_workers
        self.subset_shuffle_every_epochs = subset_shuffle_every_epochs
        self.subset_seed = int(subset_seed)

        self._subset_perm = None
        self._subset_perm_cycle_index = None
        self._subset_perm_size = None

    def get_batch_sampler(self, dataset: Dataset, shuffle: bool):
        from delpi.utils.batch_sampler import get_batch_sampler_for_seq_data

        return get_batch_sampler_for_seq_data(
            dataset,
            batch_grouping_column="seq_len",
            world_size=self.trainer.world_size,
            shuffle=shuffle,
            rand_seed=self.current_epoch,
            batch_size=self.batch_size,
            local_rank=self.local_rank,
        )

    def train_dataloader(self):
        """Create training data loader."""
        dataset = self.get_trainset()
        batch_sampler = self.get_batch_sampler(dataset, shuffle=True)

        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        dataset = self.val_ds
        batch_sampler = self.get_batch_sampler(dataset, shuffle=False)

        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
