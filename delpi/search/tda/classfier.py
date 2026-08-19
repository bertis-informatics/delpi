import torch
import torch.nn as nn
import torchmetrics

import tqdm
import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from delpi.model.focal_loss import AsymmetricFocalLoss
from delpi.utils.metric import RecallAtFDR
from delpi.utils.scheduler import get_cosine_schedule_with_warmup


class TargetDecoyClassifier(pl.LightningModule):

    def __init__(
        self,
        input_size,
        layers=[64, 32],
        dropout=0.1,
        num_warmup_steps=3,
        num_training_steps=30,
        max_lr=1e-3,
        focal_loss_gamma_pos=0.0,
        focal_loss_gamma_neg=0.0,
        focal_loss_clip=0.05,
        weight_decay=0.05,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        fc_layers = [nn.LayerNorm(input_size, eps=1e-8)]
        num_neurons = [input_size] + layers
        for i in range(len(num_neurons) - 1):
            fc_layers.append(nn.Linear(num_neurons[i], num_neurons[i + 1]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
        fc_layers.append(nn.Linear(num_neurons[-1], 1))

        self.classifier = nn.Sequential(*fc_layers)

        self.train_auc = torchmetrics.AUROC(task="binary")
        self.valid_auc = torchmetrics.AUROC(task="binary")
        self.train_ap = torchmetrics.AveragePrecision(task="binary")
        self.valid_ap = torchmetrics.AveragePrecision(task="binary")
        self.valid_rc = RecallAtFDR(fdr_cutoff=0.01)
        if focal_loss_gamma_pos == 0.0 and focal_loss_gamma_neg == 0.0:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = AsymmetricFocalLoss(
                focal_loss_gamma_pos, focal_loss_gamma_neg, focal_loss_clip
            )
        # self.criterion = NURiskLoss(pi_p=0.1)
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.weight_decay = weight_decay
        self.max_lr = max_lr
        self.save_hyperparameters()

    def forward(self, X):
        logits = self.classifier(X)
        return logits

    def _compute_loss(self, X, y_true, batch_idx):

        y_true = y_true.to(torch.float32)
        logits = self(X)
        y_proba = torch.sigmoid(logits)
        loss = self.criterion(logits, y_true)

        return loss, y_true, y_proba

    def training_step(self, batch, batch_idx):

        X, y_true = batch
        batch_size = X.shape[0]
        loss, y_true, y_pred = self._compute_loss(X, y_true, batch_idx)

        # batch_size = len(y_pred)
        self.train_auc.update(y_pred, y_true)
        self.train_ap.update(y_pred, y_true.to(torch.int32))

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
            "train_auc",
            self.train_auc,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_ap",
            self.train_ap,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):

        X, y_true = batch
        batch_size = X.shape[0]
        loss, y_true, y_pred = self._compute_loss(X, y_true, batch_idx)

        batch_size = len(y_pred)
        self.valid_auc.update(y_pred, y_true)
        self.valid_ap.update(y_pred, y_true.to(torch.int32))
        self.valid_rc.update(y_pred, y_true)

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
            "val_auc",
            self.valid_auc,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val_ap", self.valid_ap, on_step=False, on_epoch=True, batch_size=batch_size
        )
        self.log(
            "val_recall",
            self.valid_rc,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.max_lr, weight_decay=self.weight_decay
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        return ({"optimizer": optimizer, "lr_scheduler": scheduler},)

    def compute_score(
        self,
        feature_arr: np.ndarray,
        batch_size: int = 512,
    ) -> np.ndarray:

        self.eval()
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(feature_arr)),
            shuffle=False,
            batch_size=batch_size,
        )
        logit_arr = np.empty(len(feature_arr), dtype=np.float32)
        st = 0
        with torch.inference_mode():
            for batch in tqdm.tqdm(test_loader, desc="Scoring"):
                x_batch = batch[0]
                logits = self(x_batch.to(self.device))
                logits = logits.flatten().detach().cpu().numpy()
                ed = st + logits.shape[0]
                logit_arr[st:ed] = logits
                st = ed

        return logit_arr
