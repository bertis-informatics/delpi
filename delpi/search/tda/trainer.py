import logging
import math
from pathlib import Path

import torch

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from delpi.constants import PMSM_EMBEDDING_DIM
from delpi.search.tda.classfier import TargetDecoyClassifier
from delpi.search.tda.features import FEATURE_DIM
from delpi.utils.down_sampler import DownsampleSampler

logger = logging.getLogger(__name__)

DEFAULT_TRAINING_PARAMS = {
    "min_epochs": 10,
    "max_epochs": 40,
    "num_warmup_steps": 4,
    "random_seed": 928,
    # upper bound; the batch size follows the epoch size (_resolve_batch_size)
    "batch_size": 2048,
    "min_batch_size": 512,
    "min_steps_per_epoch": 1024,
    "train_split": 0.8,
    "max_val_samples": 2**17,
    "max_train_samples_per_epoch": 2**21,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-4,
}

DEFAULT_MODEL_PARAMS = {
    # PmSM embedding followed by the rescoring features (tda/features.py)
    "input_size": FEATURE_DIM,
    "embedding_dim": PMSM_EMBEDDING_DIM,
    "layers": [64, 32],
    "dropout": 0,
    "focal_loss_gamma_pos": 0.0,
    "focal_loss_gamma_neg": 0.0,
    "focal_loss_clip": 0,
}

# max share of free GPU memory for the training tensors
_GPU_DATA_MEMORY_FRACTION = 0.5


class TensorBatchLoader:
    """Batches sliced from in-memory tensors on their device.

    With `n`, each epoch draws a new random subset of at most `n` rows (like
    `DownsampleSampler`); otherwise rows are yielded in order.
    """

    def __init__(
        self,
        tensors: tuple[torch.Tensor, ...],
        batch_size: int,
        n: int | None = None,
        seed: int = 0,
    ):
        self.tensors = tensors
        self.batch_size = batch_size
        self.n_rows = tensors[0].shape[0]
        self.shuffle = n is not None
        self.n = self.n_rows if n is None else min(n, self.n_rows)
        self.generator = (
            torch.Generator(device=tensors[0].device).manual_seed(seed)
            if self.shuffle
            else None
        )

    def __len__(self) -> int:
        return math.ceil(self.n / self.batch_size)

    def __iter__(self):
        bs = self.batch_size
        if not self.shuffle:
            for start in range(0, self.n, bs):
                yield tuple(t[start : start + bs] for t in self.tensors)
            return
        index = torch.randperm(
            self.n_rows, generator=self.generator, device=self.generator.device
        )[: self.n]
        for start in range(0, self.n, bs):
            batch_index = index[start : start + bs]
            yield tuple(t[batch_index] for t in self.tensors)


def _resolve_batch_size(
    n_epoch_samples: int,
    max_batch_size: int,
    min_batch_size: int,
    min_steps_per_epoch: int,
) -> int:
    """Largest power of two giving at least `min_steps_per_epoch` steps per
    epoch, clamped to [min_batch_size, max_batch_size]."""
    target = n_epoch_samples // min_steps_per_epoch
    power_of_two = 1 << (target.bit_length() - 1) if target > 0 else 1
    return min(max_batch_size, max(min_batch_size, power_of_two))


def _tensor_device(device: torch.device, tensors) -> torch.device:
    """`device` when it is a GPU with room for `tensors`, else the CPU."""
    if device.type != "cuda":
        return torch.device("cpu")
    n_bytes = sum(t.element_size() * t.nelement() for t in tensors)
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if n_bytes <= _GPU_DATA_MEMORY_FRACTION * free_bytes:
        return device
    logger.warning(
        f"Rescoring data ({n_bytes / 2**30:.1f} GiB) does not fit in free GPU "
        f"memory ({free_bytes / 2**30:.1f} GiB); slicing batches on the CPU"
    )
    return torch.device("cpu")


class TargetDecoyTrainer:
    """Trains target-decoy classifiers."""

    def __init__(self, model_params: dict = None, training_params: dict = None):
        self.model_params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
        self.training_params = {**DEFAULT_TRAINING_PARAMS, **(training_params or {})}
        self.best_model_path = None

    def train(
        self,
        model_version: str,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset,
        output_dir: Path,
        device: torch.device,
    ):
        """Train a classifier and set `best_model_path` (lowest val_loss).

        `val_dataset` should be group-disjoint from `train_dataset`. CSV logs
        and the checkpoint go under `output_dir`, versioned by `model_version`.
        """
        batch_size = self._batch_size(model_version, len(train_dataset))

        if isinstance(train_dataset, TensorDataset) and isinstance(
            val_dataset, TensorDataset
        ):
            # copy the tensors to the GPU once and slice batches there
            data_device = _tensor_device(
                device, train_dataset.tensors + val_dataset.tensors
            )
            train_loader = TensorBatchLoader(
                tuple(t.to(data_device) for t in train_dataset.tensors),
                batch_size=batch_size,
                n=self.training_params["max_train_samples_per_epoch"],
                seed=self.training_params["random_seed"],
            )
            val_loader = TensorBatchLoader(
                tuple(t.to(data_device) for t in val_dataset.tensors),
                batch_size=batch_size,
            )
        else:
            num_workers = (
                0
                if isinstance(train_dataset, TensorDataset) or train_dataset.use_memory
                else 4
            )

            sampler = DownsampleSampler(
                train_dataset,
                n=self.training_params["max_train_samples_per_epoch"],
                seed=self.training_params["random_seed"],
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                persistent_workers=(num_workers > 0),
            )
            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=(num_workers > 0),
            )

        model = self._create_model()

        callbacks = self._setup_callbacks()
        logger = CSVLogger(save_dir=output_dir, version=model_version)

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
        self.best_model_path = callbacks[1].best_model_path

        if isinstance(train_loader, TensorBatchLoader):
            # free the device copies; the Trainer still references the loaders
            train_loader.tensors = val_loader.tensors = ()

    def _batch_size(self, model_version: str, n_train: int) -> int:
        params = self.training_params
        n_epoch_samples = min(params["max_train_samples_per_epoch"], n_train)
        batch_size = _resolve_batch_size(
            n_epoch_samples,
            max_batch_size=params["batch_size"],
            min_batch_size=params["min_batch_size"],
            min_steps_per_epoch=params["min_steps_per_epoch"],
        )
        logger.info(
            f"Rescoring model {model_version}: batch size {batch_size} "
            f"({n_epoch_samples:,} samples per epoch)"
        )
        return batch_size

    def get_best_model(self) -> TargetDecoyClassifier:
        return TargetDecoyClassifier.load_from_checkpoint(
            self.best_model_path,
        ).eval()

    def _create_model(self) -> TargetDecoyClassifier:
        """Create a target-decoy classifier model."""
        torch.manual_seed(self.training_params["random_seed"])
        return TargetDecoyClassifier(
            input_size=self.model_params["input_size"],
            embedding_dim=self.model_params["embedding_dim"],
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
            filename="val_best",
            # filename="{epoch}-{val_loss:.4f}",
        )
        return [early_stop_callback, checkpoint_callback]
