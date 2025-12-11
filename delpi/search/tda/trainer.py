from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import tqdm

from torch.utils.data import DataLoader, TensorDataset, random_split
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split

from delpi.search.tda.classfier import TargetDecoyClassifier
from delpi.utils.down_sampler import DownsampleSampler
from delpi.search.tda.dataset import PMSMDataset


# Default training parameters
DEFAULT_TRAINING_PARAMS = {
    "min_epochs": 10,
    "max_epochs": 40,
    "num_warmup_steps": 4,
    "random_seed": 928,
    "batch_size": 512,
    "train_split": 0.8,
    "max_val_samples": 120000,
    "max_train_samples_per_epoch": 2000000,
    "test_split": 0.5,  # only half of target peptides are used for training
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-6,
}

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    "layers": [64, 32],
    "dropout": 0,
    "focal_loss_gamma_pos": 0.0,
    "focal_loss_gamma_neg": 4.0,
    "focal_loss_clip": 0,
}


class TargetDecoyTrainer:
    """Handles training of target-decoy classifiers."""

    def __init__(self, model_params: dict = None, training_params: dict = None):
        self.model_params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
        self.training_params = {**DEFAULT_TRAINING_PARAMS, **(training_params or {})}

    def train(
        self,
        model_version: str,
        train_dataset: Union[PMSMDataset, TensorDataset],
        test_dataset: Union[PMSMDataset, TensorDataset],
        output_dir: Path,
        device: torch.device,
    ) -> np.ndarray:
        """
        Train a target-decoy classifier and score the test dataset.

        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset for scoring
            model_log_dir: Directory to save model logs
            device: PyTorch device for training

        Returns:
            Array of scores for the test dataset
        """
        torch.set_float32_matmul_precision("medium")

        input_size = PMSMDataset.PMSM_EMBEDDING_DIM
        num_workers = (
            0
            if isinstance(train_dataset, TensorDataset) or train_dataset.use_memory
            else 4
        )

        # Split training data
        train_ds, val_ds = self._split_training_data(train_dataset)
        sampler = DownsampleSampler(
            train_ds,
            n=self.training_params["max_train_samples_per_epoch"],
            seed=self.training_params["random_seed"],
        )

        # Create data loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.training_params["batch_size"],
            sampler=sampler,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            batch_size=self.training_params["batch_size"],
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

        # Create model
        model = self._create_model(input_size)

        # Setup callbacks
        callbacks = self._setup_callbacks()
        logger = CSVLogger(save_dir=output_dir, version=model_version)

        # Train model
        trainer = Trainer(
            min_epochs=self.training_params["min_epochs"],
            max_epochs=self.training_params["max_epochs"],
            accelerator=device.type,
            devices=[device.index] if device.index is not None else [0],
            logger=logger,
            default_root_dir=logger.log_dir,
            callbacks=callbacks,
            enable_model_summary=False,
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        # Score test dataset
        return self._score_dataset(test_dataset, callbacks[1].best_model_path, device)

    def _split_training_data(
        self, train_dataset: Union[PMSMDataset, TensorDataset]
    ) -> Union[Tuple[PMSMDataset, PMSMDataset], Tuple[TensorDataset, TensorDataset]]:
        """Split training dataset into train and validation sets."""

        train_split = self.training_params["train_split"]
        n_trains = len(train_dataset)
        if n_trains * (1 - train_split) > self.training_params["max_val_samples"]:
            split_lens = [
                n_trains - self.training_params["max_val_samples"],
                self.training_params["max_val_samples"],
            ]
        else:
            split_lens = [train_split, 1 - train_split]

        if isinstance(train_dataset, TensorDataset):
            generator = torch.Generator()
            if self.training_params["random_seed"] is not None:
                generator = generator.manual_seed(self.training_params["random_seed"])
            return random_split(train_dataset, split_lens, generator=generator)

        train_df, val_df = train_test_split(
            train_dataset.pmsm_df,
            test_size=split_lens[-1],
            shuffle=True,
            random_state=self.training_params["random_seed"],
        )

        train_ds = PMSMDataset(
            train_df,
            hdf_files=train_dataset.hdf_files,
            hdf_group_key=train_dataset.group_key,
            rt_scale=train_dataset.rt_scale,
            data_dict=train_dataset.data_dict,
        )

        val_ds = PMSMDataset(
            val_df,
            hdf_files=train_dataset.hdf_files,
            hdf_group_key=train_dataset.group_key,
            rt_scale=train_dataset.rt_scale,
            data_dict=train_dataset.data_dict,
        )

        return train_ds, val_ds

    def _create_model(self, input_size: int) -> TargetDecoyClassifier:
        """Create a target-decoy classifier model."""
        torch.manual_seed(self.training_params["random_seed"])
        return TargetDecoyClassifier(
            input_size=input_size,
            layers=self.model_params["layers"],
            dropout=self.model_params["dropout"],
            focal_loss_gamma_pos=self.model_params["focal_loss_gamma_pos"],
            focal_loss_gamma_neg=self.model_params["focal_loss_gamma_neg"],
            focal_loss_clip=self.model_params["focal_loss_clip"],
            num_warmup_steps=self.training_params["num_warmup_steps"],
            num_training_steps=self.training_params["max_epochs"],
        )

    def _setup_callbacks(self) -> list:
        """Setup training callbacks."""
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=self.training_params["early_stopping_patience"],
            min_delta=self.training_params["early_stopping_min_delta"],
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="{epoch}-{val_loss:.4f}",
        )
        return [early_stop_callback, checkpoint_callback]

    def _score_dataset(
        self, test_dataset: TensorDataset, model_path: str, device: torch.device
    ) -> np.ndarray:
        """Score a dataset using the trained model."""
        test_loader = DataLoader(
            test_dataset, shuffle=False, batch_size=self.training_params["batch_size"]
        )

        trained_model = TargetDecoyClassifier.load_from_checkpoint(
            model_path, map_location=device
        ).eval()

        test_score_arr = np.empty(len(test_dataset), dtype=np.float32)

        st = 0
        with torch.inference_mode():
            for batch in tqdm.tqdm(test_loader, desc="Scoring test dataset"):
                X, y_true = batch
                logits = trained_model(X.to(device))
                logits = logits.flatten().detach().cpu().numpy()
                ed = st + logits.shape[0]
                test_score_arr[st:ed] = logits
                st = ed

        return test_score_arr
