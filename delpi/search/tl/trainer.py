from pathlib import Path

import numpy as np
import torch

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split

from delpi.model.spec_lib.ms2_predictor import Ms2SpectrumPredictor
from delpi.search.tl.dataset import TransferLearningDataset, LABEL_DTYPE
from delpi.search.result_aggregator import ResultsAggregator
from delpi import MODEL_DIR

# Default training parameters
DEFAULT_TRAINING_PARAMS = {
    "max_epochs": 20,
    "num_warmup_steps": 10,
    "random_seed": 928,
    "batch_size": 512,
    "train_split": 0.8,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-5,
    "max_lr": 1e-4,
    "num_workers": 8,
}


class TransferLearningTrainer:

    def __init__(self, training_params: dict = None):
        self.training_params = {**DEFAULT_TRAINING_PARAMS, **(training_params or {})}

    def train(
        self,
        output_dir: Path,
        result_aggregator: ResultsAggregator,
        device: torch.device,
    ) -> np.ndarray:

        logger = CSVLogger(save_dir=output_dir, version=f"ms2_predictor_tl")
        hdf_files = result_aggregator.get_hdf_files()
        label_df = result_aggregator.get_tl_label_df()
        labels = np.empty(len(label_df), dtype=LABEL_DTYPE)
        for field in LABEL_DTYPE.names:
            labels[field] = label_df[field].to_numpy()

        in_memory = len(labels) < 1_000_000
        data_dict = (
            result_aggregator.get_tl_data(
                data_keys=["x_aa", "x_mod", "x_meta", "x_intensity"]
            )
            if in_memory
            else None
        )

        val_size = min(int(len(labels) * 0.2), 10000)
        train_labels, val_labels = train_test_split(
            labels, test_size=val_size, random_state=718, shuffle=True
        )

        train_ds = TransferLearningDataset(hdf_files, train_labels, data_dict=data_dict)
        val_ds = TransferLearningDataset(hdf_files, val_labels, data_dict=data_dict)
        max_epochs = self.training_params["max_epochs"]
        fraction = (
            None if len(train_labels) < 100000 else min(100000 / len(train_labels), 0.5)
        )

        model = Ms2SpectrumPredictor(
            encoder_type="transformer",
            mod_embedding_dim=8,
            meta_embedding_dim=4,
            embedding_dim=192,
            dropout=0.1,
            max_lr=self.training_params["max_lr"],
            num_warmup_steps=self.training_params["num_warmup_steps"],
            num_training_steps=max_epochs,
            transformer_depth=12,
            transformer_num_heads=12,
            transformer_qkv_bias=True,
            transformer_drop_path_rate=0.1,
            layer_decay=0.9,
        )

        pretrained_weights = Ms2SpectrumPredictor.load(
            MODEL_DIR / "delpi.ms2_predictor.pth"
        ).state_dict()
        _ = model.load_state_dict(pretrained_weights, strict=False)

        # Set datasets for multi-GPU training
        model.set_dataset(
            train_ds,
            val_ds,
            batch_size=self.training_params["batch_size"],
            num_workers=0 if in_memory else self.training_params["num_workers"],
            fractions=fraction,
        )
        # Setup callbacks
        callbacks = self._setup_callbacks()

        # Setup trainer
        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator=device.type,
            devices=[device.index] if device.index is not None else [0],
            logger=logger,
            default_root_dir=logger.log_dir,
            callbacks=callbacks,
            enable_model_summary=False,
        )

        # Train model
        trainer.fit(model=model)
        trained_model = (
            Ms2SpectrumPredictor.load_from_checkpoint(callbacks[1].best_model_path)
            .to(device)
            .eval()
        )

        return trained_model

    def _setup_callbacks(self) -> list:
        """Setup training callbacks."""
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=self.training_params["early_stopping_patience"],
            min_delta=self.training_params["early_stopping_min_delta"],
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            filename="{epoch}",
        )
        return [early_stop_callback, checkpoint_callback]


def test():
    trainer = TransferLearningTrainer()
    self = trainer
    # output_dir = Path(r"/data1/benchmark/DIA/2024-HAP1/delpi")
    output_dir = Path(r"/data1/MassSpecData/DIA_LIBD/delpi")
    device = torch.device("cuda:0")
    model = trainer.train(
        output_dir=output_dir,
        device=device,
    )

    # dataset = PmsmDataset(pmsm_df, nce=30, fragmentation=0, mass_analyzer=0)

    "precursor_index", "peptidoform_index",
    "peptide_index", "sequence_length"

    # ms2_df = self.predict_ms2_spectra(
    #         peptide_df=peptide_df,
    #         modification_df=modification_df,
    #         precursor_df=precursor_df,
    #         prefix_mass_container=prefix_mass_container,
    #         batch_size=512,
    #         detectable_min_mz=min_fragment_mz,
    #         detectable_max_mz=max_fragment_mz,
    #     )
