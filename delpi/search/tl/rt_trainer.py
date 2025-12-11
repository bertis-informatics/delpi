from pathlib import Path

import polars as pl
import numpy as np
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split

from delpi.model.spec_lib.rt_predictor import RetentionTimePredictor
from delpi.search.tl.dataset import TransferLearningDatasetForRT
from delpi.search.result_aggregator import ResultsAggregator
from delpi.model.rt_calibrator import RetentionTimeCalibrator
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


def make_rt_df(run_data_dict):
    rt_df = pl.concat(
        (
            pl.DataFrame({k: d[k] for k in ["precursor_index", "x_rt"]})
            for k, d in run_data_dict.items()
        ),
        how="vertical",
    )
    return rt_df


class TransferLearningTrainerForRT:

    def __init__(self, training_params: dict = None):
        self.training_params = {**DEFAULT_TRAINING_PARAMS, **(training_params or {})}

    def train(
        self,
        output_dir: Path,
        result_aggregator: ResultsAggregator,
        device: torch.device,
    ) -> np.ndarray:

        torch.set_float32_matmul_precision("medium")
        logger = CSVLogger(save_dir=output_dir, version=f"rt_predictor_tl")

        label_df = result_aggregator.get_tl_label_df()
        data_dict = result_aggregator.get_tl_data(
            data_keys=["precursor_index", "x_aa", "x_mod", "x_rt"]
        )

        if len(data_dict) > 1:
            data_dict = self._align_retention_times(data_dict)

        train_df, val_df = train_test_split(
            label_df,
            test_size=self.training_params["val_split"],
            random_state=self.training_params["random_seed"],
            shuffle=True,
        )

        ## to avoid too-long training time, limit the number of samples
        if train_df.shape[0] > self.training_params["max_train_samples"]:
            train_df = train_df.sample(
                n=self.training_params["max_train_samples"],
                seed=self.training_params["random_seed"],
            )
        if val_df.shape[0] > self.training_params["max_val_samples"]:
            val_df = val_df.sample(
                n=self.training_params["max_val_samples"],
                seed=self.training_params["random_seed"],
            )

        train_ds = TransferLearningDatasetForRT(train_df, data_dict=data_dict)
        val_ds = TransferLearningDatasetForRT(val_df, data_dict=data_dict)

        pretrained_weights = torch.load(
            MODEL_DIR / "delpi.rt_predictor.pth",
            weights_only=False,
            map_location=device,
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

    def _align_retention_times(self, data_dict):
        # Align retention times before training, setting the first run as reference
        if len(data_dict) > 1:
            ref_rt_df = make_rt_df(data_dict[0])
            for i in range(1, len(data_dict)):
                rt_df = make_rt_df(data_dict[i])
                aligner = RetentionTimeCalibrator.train_aligner(
                    ref_rt_df, rt_df, degree=2
                )
                # align retention times and update in data_dict
                for k, v in data_dict[i].items():
                    v["x_rt"] = aligner.predict(v["x_rt"].reshape((-1, 1))).flatten()

        return data_dict


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
