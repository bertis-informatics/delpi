from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split

from delpi.model.spec_lib.rt_predictor import RetentionTimePredictor
from delpi.search.tl.dataset import TransferLearningDatasetForRT, LABEL_DTYPE
from delpi.search.result_aggregator import ResultsAggregator
from delpi import MODEL_DIR

# Default training parameters
DEFAULT_TRAINING_PARAMS = {
    "max_epochs": 40,
    "num_warmup_steps": 4,
    "random_seed": 928,
    "batch_size": 512,
    "val_split": 0.2,
    "max_val_samples": 200000,
    "max_train_samples": 1000000,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-1,
    "max_lr": 1e-3,
    "num_workers": 8,
}


class TransferLearningTrainerForRT:

    def __init__(self, training_params: dict = None):
        self.training_params = {**DEFAULT_TRAINING_PARAMS, **(training_params or {})}

    def train(
        self,
        output_dir: Path,
        result_aggregator: ResultsAggregator,
        device: torch.device,
        tl_ms2_h5_path: Path,
    ) -> np.ndarray:

        logger = CSVLogger(save_dir=output_dir, version=f"rt_predictor_tl")

        label_df = result_aggregator.get_tl_label_df(tl_ms2_h5_path)
        labels = np.empty(len(label_df), dtype=LABEL_DTYPE)
        for field in LABEL_DTYPE.names:
            labels[field] = label_df[field].to_numpy()

        data_dict = result_aggregator.get_tl_data(
            tl_ms2_h5_path, data_keys=["precursor_index", "x_aa", "x_mod", "x_rt"]
        )

        train_labels, val_labels = train_test_split(
            labels,
            test_size=self.training_params["val_split"],
            random_state=self.training_params["random_seed"],
            shuffle=True,
        )

        ## to avoid too-long training time, limit the number of samples
        rng = np.random.default_rng(self.training_params["random_seed"])
        if len(train_labels) > self.training_params["max_train_samples"]:
            idx = rng.choice(
                len(train_labels),
                self.training_params["max_train_samples"],
                replace=False,
            )
            train_labels = train_labels[idx]
        if len(val_labels) > self.training_params["max_val_samples"]:
            idx = rng.choice(
                len(val_labels), self.training_params["max_val_samples"], replace=False
            )
            val_labels = val_labels[idx]

        train_ds = TransferLearningDatasetForRT(train_labels, data_dict=data_dict)
        val_ds = TransferLearningDatasetForRT(val_labels, data_dict=data_dict)

        pretrained_weights = RetentionTimePredictor.load(
            MODEL_DIR / "delpi.rt_predictor.pth"
        ).state_dict()

        model = RetentionTimePredictor(
            encoder_type="cnn_rnn",
            aa_vocab_size=22,
            aa_embedding_dim=24,
            embedding_dim=128,
            dropout=0.1,
            num_layers=1,
            max_lr=self.training_params["max_lr"],
            num_warmup_steps=self.training_params["num_warmup_steps"],
            num_training_steps=self.training_params["max_epochs"],
            fine_tuning=True,
            seq_len_column="seq_len",
        )

        _ = model.load_state_dict(pretrained_weights, strict=False)
        model.set_dataset(
            train_ds,
            val_ds,
            batch_size=self.training_params["batch_size"],
            num_workers=0,
        )
        # Setup callbacks
        callbacks = self._setup_callbacks()

        # Setup trainer
        trainer = Trainer(
            max_epochs=self.training_params["max_epochs"],
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
            RetentionTimePredictor.load_from_checkpoint(callbacks[1].best_model_path)
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
    trainer = TransferLearningTrainerForRT()
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
