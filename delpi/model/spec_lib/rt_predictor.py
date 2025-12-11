from typing import Dict

import polars as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchmetrics
from lightning.pytorch import LightningModule

from delpi.model.spec_lib.block import ResNet1D, BiLSTM, Transformer, Permute
from delpi.model.pos_encoder import PositionalEncoding
from delpi.search.tl.lr_decay import param_groups_lrd_for_rt_predictor
from delpi.utils.scheduler import get_cosine_schedule_with_warmup
from delpi.model.spec_lib.aa_encoder import MOD_FEATURE_MAP


class RetentionTimePredictor(LightningModule):

    def __init__(
        self,
        aa_vocab_size: int = 22,
        aa_embedding_dim: int = 32,
        mod_embedding_dim: int = 8,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 1,
        max_lr: float = 1e-3,
        num_warmup_steps: int = 4,
        num_training_steps: int = 40,
        encoder_type: str = "cnn_rnn",  # "cnn_rnn" or "transformer"
        # Transformer specific parameters
        transformer_depth: int = 2,
        transformer_num_heads: int = 4,
        transformer_qkv_bias: bool = True,
        transformer_drop_path_rate: float = 0.0,
        fine_tuning: bool = False,
        seq_len_column: str = "peptide_length",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.encoder_type = encoder_type

        # Embedding layer
        self.aa_embedding = nn.Embedding(
            aa_vocab_size, aa_embedding_dim - mod_embedding_dim
        )
        # self.mod_embedding = ModificationEncoder(embedding_dim=mod_embedding_dim)
        self.mod_embedding = nn.Linear(MOD_FEATURE_MAP.shape[-1], mod_embedding_dim)

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
                    return_sequences=False,  # Return final hidden state for RT
                ),
            )
            encoder_output_dim = num_layers * 2 * embedding_dim  # Bidirectional LSTM

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
                    return_sequences=False,  # Return CLS token for RT
                ),
            )
            encoder_output_dim = embedding_dim

        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. Choose 'cnn_rnn' or 'transformer'"
            )

        # Regression head
        self.rt_predictor = torch.nn.Sequential(
            nn.Linear(encoder_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.train_corr = torchmetrics.PearsonCorrCoef()
        self.valid_corr = torchmetrics.PearsonCorrCoef()

        self.max_lr = max_lr
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.fine_tuning = fine_tuning
        self.seq_len_column = seq_len_column

        self.save_hyperparameters()

    def forward(self, x_aa, x_mod):
        """
        Forward pass for RT prediction.

        Args:
            x_aa: Amino acid sequence tensor (B, L+2) where L is peptide length
            x_mod: Modification tensor (B, max_mod_count, 2) with modification tuples

        Returns:
            Retention time predictions tensor (B, 1)
        """
        # Pass through AA and Mod embedding layers

        x_aa_emb = self.aa_embedding(x_aa.to(torch.int32))
        # x_mod_emb = self.mod_embedding(x_mod, x_aa_emb.shape[1])
        x_mod_emb = self.mod_embedding(x_mod)
        x_emb = torch.cat([x_aa_emb, x_mod_emb], dim=-1)

        # Pass through encoder (both CNN+RNN and Transformer are now Sequential)
        x_emb = self.encoder(x_emb)

        # Predict retention time
        y_pred = self.rt_predictor(x_emb)

        return y_pred

    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> pl.DataFrame:

        peptidoform_index_arr = batch["peptidoform_index"].to(torch.uint32).numpy()
        x_aa = batch["x_aa"].to(device=self.device)
        x_mod = batch["x_mod"].to(device=self.device)
        y_pred = self(x_aa, x_mod)

        return pl.DataFrame(
            {
                "peptidoform_index": peptidoform_index_arr,
                "ref_rt": y_pred.flatten().detach().cpu().numpy(),
            }
        )

    def _compute_loss(self, x_aa, x_mod, y_true):

        y_pred = self(x_aa, x_mod)
        # loss = nn.functional.mse_loss(y_pred, y_true)
        loss = nn.functional.huber_loss(y_pred, y_true, delta=10)

        return loss, y_true, y_pred

    def training_step(self, batch, batch_idx):

        x_aa = batch["x_aa"]
        x_mod = batch["x_mod"]
        y_true = batch["rt"]
        batch_size = len(y_true)
        loss, y_true, y_pred = self._compute_loss(x_aa, x_mod, y_true)

        self.train_corr.update(y_pred, y_true)

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

        return loss

    def validation_step(self, batch, batch_idx):

        x_aa = batch["x_aa"]
        x_mod = batch["x_mod"]
        y_true = batch["rt"]
        batch_size = len(y_true)
        loss, y_true, y_pred = self._compute_loss(x_aa, x_mod, y_true)

        self.valid_corr.update(y_pred, y_true)

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

        return loss

    def configure_optimizers(self):

        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.max_lr)
        if self.fine_tuning:
            param_groups = param_groups_lrd_for_rt_predictor(
                model=self, weight_decay=0.05, layer_decay=0.5
            )
        else:
            param_groups = self.parameters()

        optimizer = torch.optim.AdamW(param_groups, lr=self.max_lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        return ({"optimizer": optimizer, "lr_scheduler": scheduler},)

    def set_dataset(
        self,
        train_dataset,
        val_dataset,
        batch_size,
        num_workers=8,
    ):
        """Set datasets for training and validation."""
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_batch_sampler(self, dataset: Dataset, shuffle: bool):
        from delpi.utils.batch_sampler import get_batch_sampler_for_seq_data

        return get_batch_sampler_for_seq_data(
            dataset,
            batch_grouping_column=self.seq_len_column,
            world_size=self.trainer.world_size,
            shuffle=shuffle,
            rand_seed=self.current_epoch,
            batch_size=self.batch_size,
            local_rank=self.local_rank,
        )

    def train_dataloader(self):
        """Create training data loader."""
        dataset = self.train_ds
        # dataset = self.train_ds.__class__(self.train_ds.label_df.clone())
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
