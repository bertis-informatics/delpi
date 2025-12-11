from typing import Dict, List

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
import torchmetrics
from torch.utils.data import DataLoader, Dataset

from delpi.model.spec_lib.block import ResNet1D, BiLSTM, Transformer, Permute
from delpi.model.pos_encoder import PositionalEncoding
from delpi.utils.metric import SpectralAngle
from delpi.search.tl.lr_decay import param_groups_lrd
from delpi.utils.scheduler import get_cosine_schedule_with_warmup
from delpi.model.spec_lib.aa_encoder import MOD_FEATURE_MAP

EPS = 1e-9
FRAG_TYPE_LIST = [
    "b_z1",
    "b_z2",
    "y_z1",
    "y_z2",
    "b_modloss_z1",
    "b_modloss_z2",
    "y_modloss_z1",
    "y_modloss_z2",
]


class Ms2SpectrumPredictor(LightningModule):

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
        fine_tuning: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

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
        # Input: (B, L-2, encoder_output_dim) -> Output: (B, L-2, 8)
        self.fragment_predictor = nn.Sequential(
            nn.Linear(encoder_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 8),
            nn.ReLU(),  # Ensure non-negative intensities
        )

        # Metrics
        self.train_corr = torchmetrics.PearsonCorrCoef()
        self.valid_corr = torchmetrics.PearsonCorrCoef()
        self.train_sa = SpectralAngle()
        self.valid_sa = SpectralAngle()

        self.max_lr = max_lr
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.fine_tuning = fine_tuning

        self.save_hyperparameters()

    def forward(self, x_aa, x_mod, x_meta):
        """
        Forward pass for MS2 spectrum prediction.

        Args:
            x_aa: Amino acid sequence tensor (B, L+2) where L is peptide length
            x_mod: Modification tensor (B, max_mod_count, 2) with modification tuples

        Returns:
            Fragment intensities tensor (B, L-2, 8)
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

        # Remove terminal residues: [B, L+2, D] -> [B, L-2, D]
        # This assumes the first and last positions are terminal residues
        x_emb = x_emb[:, 2:-1, :]

        # Predict fragment intensities
        y_pred = self.fragment_predictor(x_emb)

        return y_pred

    def predict_batch_arr(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """_summary_

        Returns:
            pl.DataFrame: _description_
            ┌──────────┬──────────┬──────────┬──────────┬─────────────────┬────────────────┐
            │ b_z1     ┆ y_z1     ┆ b_z2     ┆ y_z2     ┆ precursor_index ┆ cleavage_index │
            │ ---      ┆ ---      ┆ ---      ┆ ---      ┆ ---             ┆ ---            │
            │ f32      ┆ f32      ┆ f32      ┆ f32      ┆ i64             ┆ u8             │
            ╞══════════╪══════════╪══════════╪══════════╪═════════════════╪════════════════╡
            │ 0.0      ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ 7               ┆ 0              │
            │ 0.331823 ┆ 0.0      ┆ 0.0      ┆ 0.024395 ┆ 7               ┆ 1              │
            │ …        ┆ …        ┆ …        ┆ …        ┆ …               ┆ …              │
            │ 0.016013 ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ 1761            ┆ 10             │
            └──────────┴──────────┴──────────┴──────────┴─────────────────┴────────────────┘
        """
        precursor_index_arr = batch["precursor_index"].to(torch.uint32).numpy()
        x_aa = batch["x_aa"].to(device=self.device)
        x_mod = batch["x_mod"].to(device=self.device)
        x_meta = batch["x_meta"].to(device=self.device)

        y_pred = self(x_aa, x_mod, x_meta)

        # if include_modloss:
        #     ion_type_count = 8
        # else:
        #     ion_type_count = 4
        #     y_pred = y_pred[..., :ion_type_count]
        # scale = torch.amax(y_pred, dim=(1, 2), keepdim=True)
        # y_pred = y_pred / (scale + EPS)

        y_pred = y_pred.detach().cpu().numpy()

        return precursor_index_arr, y_pred

    def predict_batch(
        self, batch: Dict[str, torch.Tensor], include_modloss: bool = False
    ) -> pl.DataFrame:
        """_summary_

        Returns:
            pl.DataFrame: _description_
            ┌──────────┬──────────┬──────────┬──────────┬─────────────────┬────────────────┐
            │ b_z1     ┆ y_z1     ┆ b_z2     ┆ y_z2     ┆ precursor_index ┆ cleavage_index │
            │ ---      ┆ ---      ┆ ---      ┆ ---      ┆ ---             ┆ ---            │
            │ f32      ┆ f32      ┆ f32      ┆ f32      ┆ i64             ┆ u8             │
            ╞══════════╪══════════╪══════════╪══════════╪═════════════════╪════════════════╡
            │ 0.0      ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ 7               ┆ 0              │
            │ 0.331823 ┆ 0.0      ┆ 0.0      ┆ 0.024395 ┆ 7               ┆ 1              │
            │ …        ┆ …        ┆ …        ┆ …        ┆ …               ┆ …              │
            │ 0.016013 ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ 1761            ┆ 10             │
            └──────────┴──────────┴──────────┴──────────┴─────────────────┴────────────────┘
        """

        precursor_index_arr = batch["precursor_index"].to(torch.uint32).numpy()
        x_aa = batch["x_aa"].to(device=self.device)
        x_mod = batch["x_mod"].to(device=self.device)
        x_meta = batch["x_meta"].to(device=self.device)
        cleavage_count = x_aa.shape[-1] - 3

        y_pred = self(x_aa, x_mod, x_meta)
        if include_modloss:
            ion_type_count = 8
        else:
            ion_type_count = 4
            y_pred = y_pred[..., :ion_type_count]

        scale = torch.amax(y_pred, dim=(1, 2), keepdim=True)
        y_pred = y_pred / (scale + EPS)
        y_pred = y_pred.detach().cpu().numpy()
        ion_type_count = y_pred.shape[-1]

        batch_ms2_df = pl.from_numpy(
            y_pred.reshape(-1, ion_type_count),
            schema=FRAG_TYPE_LIST[:ion_type_count],
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
            pl.col(*FRAG_TYPE_LIST[:ion_type_count]),
        )

        return batch_ms2_df

    def _compute_loss(self, x_aa, x_mod, x_meta, y_true):
        """
        Compute loss for MS2 spectrum prediction.

        Args:
            x_aa: Amino acid sequence tensor
            x_mod: Modification tensor
            y_true: True fragment intensities (B, L-2, 8)

        Returns:
            loss, y_true, y_pred
        """
        y_pred = self(x_aa, x_mod, x_meta)
        # if y_true.size(-1) < y_pred.size(-1):
        #     y_true = F.pad(
        #         y_true, (0, y_pred.size(-1) - y_true.size(-1)), "constant", 0
        #     )
        y_pred = y_pred[..., : y_true.size(-1)]

        # Use MSE loss for intensity prediction
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

        # Flatten for correlation calculation
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        self.valid_corr.update(y_pred_flat, y_true_flat)
        self.valid_sa.update(y_pred_flat, y_true_flat)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "val_corr",
            self.valid_corr,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val_sa",
            self.valid_sa,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.fine_tuning:
            param_groups = param_groups_lrd(
                model=self, weight_decay=0.05, layer_decay=0.75
            )
        else:
            param_groups = self.parameters()

        optimizer = torch.optim.AdamW(param_groups, lr=self.max_lr, weight_decay=0.05)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        return ({"optimizer": optimizer, "lr_scheduler": scheduler},)

    def get_trainset(self):
        if self.fractions is not None:
            return self.original_train_ds.make_subset(
                fractions=self.fractions,
                seed=self.current_epoch,
            )

        return self.original_train_ds

    def set_dataset(
        self,
        train_dataset,
        val_dataset,
        batch_size,
        fractions=None,
        num_workers=8,
    ):
        """Set datasets for training and validation."""
        self.original_train_ds = train_dataset
        self.val_ds = val_dataset
        self.batch_size = batch_size
        self.fractions = fractions
        self.num_workers = num_workers

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
