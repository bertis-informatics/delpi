from typing import Union

import polars as pl
import numpy as np

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from delpi.lcms.base_ion_type import BaseIonType
from delpi.lcms.fragmentation import Fragmentation
from delpi.lcms.neutral_loss import NeutralLoss
from delpi.model.spec_lib import Ms2SpectrumPredictor, RetentionTimePredictor
from delpi.model.spec_lib.dataset import PeptideDataset
from delpi.utils.batch_sampler import SeqDataBatchSampler
from delpi.database.numba.prefix_mass_array import PrefixMassArrayContainer
from delpi.database.numba.spec_lib_utils import update_speclib_arr
from delpi.utils.prefetch import Prefetcher, pin_tensor_dict
from delpi import MODEL_DIR


class SpectralLibGenerator:

    speclib_df_schema = {
        "precursor_index": pl.UInt32,
        "cleavage_index": pl.UInt8,
        "is_prefix": pl.Boolean,
        "charge": pl.UInt8,
        "mz": pl.Float32,
        "predicted_intensity": pl.Float32,
        "rank": pl.UInt8,
    }

    def __init__(
        self,
        min_charge: int = 1,
        max_charge: int = 2,
        prefix_ion_type=BaseIonType.B,
        suffix_ion_type=BaseIonType.Y,
        max_fragments=16,
        apply_phospho=False,
        device: Union[str, torch.device] = "cuda:0",
        ms2_predictor: Ms2SpectrumPredictor = None,
        rt_predictor: RetentionTimePredictor = None,
    ):
        self.max_fragments = max_fragments
        neutral_losses = [NeutralLoss.NO_LOSS]
        if apply_phospho:
            neutral_losses.append(NeutralLoss.H3O4P)

        self.fragmentation = Fragmentation(
            min_charge,
            max_charge,
            prefix_ion_type=prefix_ion_type,
            suffix_ion_type=suffix_ion_type,
            max_fragment_isotopes=1,
            neutral_losses=neutral_losses,
        )

        self.device = device
        self._ms2_predictor = ms2_predictor
        self._rt_predictor = rt_predictor

    @property
    def ms2_predictor(self):
        if self._ms2_predictor is None:
            self._ms2_predictor = torch.load(
                MODEL_DIR / "delpi.ms2_predictor.pth", weights_only=False
            )
        return self._ms2_predictor.to(self.device).eval()

    @property
    def rt_predictor(self):
        if self._rt_predictor is None:
            self._rt_predictor = torch.load(
                MODEL_DIR / "delpi.rt_predictor.pth", weights_only=False
            )
        return self._rt_predictor.to(self.device).eval()

    @property
    def param_dict(self):
        return {
            "min_charge": self.fragmentation.min_charge,
            "max_charge": self.fragmentation.max_charge,
            "prefix_ion_type": self.fragmentation.base_ion_types[0].symbol,
            "suffix_ion_type": self.fragmentation.base_ion_types[1].symbol,
            "max_fragments": self.max_fragments,
        }

    def predict_ms2_spectra(
        self,
        peptide_df: pl.DataFrame,
        modification_df: pl.DataFrame,
        precursor_df: pl.DataFrame,
        prefix_mass_container: PrefixMassArrayContainer,
        batch_size: int = 512,
        detectable_min_mz: float = 200,
        detectable_max_mz: float = 1800,
    ):
        precursor_ds = PeptideDataset(
            precursor_df, modification_df, peptide_df, level="precursor"
        )
        # include_modloss = len(self.fragmentation.neutral_losses) > 1
        batch_sampler = SeqDataBatchSampler(
            precursor_ds,
            batch_grouping_column="sequence_length",
            batch_size=batch_size,
            shuffle=False,
        )
        total = batch_sampler.count_num_of_batches()
        dl = DataLoader(
            dataset=precursor_ds, batch_sampler=batch_sampler, num_workers=0
        )

        # Pre-allocate GPU buffers for zero-allocation H2D transfer
        max_token_len = peptide_df["sequence_length"].max() + 3
        mod_feat_dim = self.ms2_predictor.mod_embedding.in_features
        X_aa_buf = torch.empty(
            (batch_size, max_token_len), dtype=torch.long, device=self.device
        )
        X_mod_buf = torch.empty(
            (batch_size, max_token_len, mod_feat_dim),
            dtype=torch.float32,
            device=self.device,
        )
        X_meta_buf = torch.empty(
            (batch_size, 4), dtype=torch.float32, device=self.device
        )

        ion_type_container = self.fragmentation.get_ion_types()
        peptidoform_index_arr = precursor_df["peptidoform_index"].to_numpy()
        max_fragments = self.max_fragments
        speclib_row_count = max_fragments * precursor_df.shape[0]

        out_precursor_index_arr = np.empty(speclib_row_count, dtype=np.uint32)
        out_clevage_index_arr = np.empty(speclib_row_count, dtype=np.uint8)
        out_is_prefix_arr = np.empty(speclib_row_count, dtype=np.bool_)
        out_charge_arr = np.empty(speclib_row_count, dtype=np.uint8)
        out_mz_arr = np.empty(speclib_row_count, dtype=np.float32)
        out_pred_intensity_arr = np.empty(speclib_row_count, dtype=np.float32)
        out_rank_arr = np.empty(speclib_row_count, dtype=np.uint8)

        with torch.inference_mode():
            for batch in tqdm(
                Prefetcher(dl, transform=pin_tensor_dict),
                total=total,
                desc="Predicting MS2 spectra",
                leave=True,
            ):
                batch_precursor_index_arr = (
                    batch["precursor_index"].to(torch.uint32).numpy()
                )
                x_aa_t = batch["x_aa"]
                x_mod_t = batch["x_mod"]
                x_meta_t = batch["x_meta"]

                n, L = x_aa_t.shape
                X_aa_buf[:n, :L].copy_(x_aa_t, non_blocking=True)
                X_mod_buf[:n, :L, :].copy_(x_mod_t, non_blocking=True)
                X_meta_buf[:n].copy_(x_meta_t, non_blocking=True)

                y_pred = self.ms2_predictor(
                    X_aa_buf[:n, :L], X_mod_buf[:n, :L, :], X_meta_buf[:n]
                )
                batch_intensity_arr = y_pred.detach().cpu().numpy()

                update_speclib_arr(
                    out_precursor_index_arr,
                    out_clevage_index_arr,
                    out_is_prefix_arr,
                    out_charge_arr,
                    out_mz_arr,
                    out_pred_intensity_arr,
                    out_rank_arr,
                    prefix_mass_container,
                    ion_type_container,
                    peptidoform_index_arr,
                    batch_precursor_index_arr,
                    batch_intensity_arr,
                    max_fragments=max_fragments,
                    detectable_min_mz=detectable_min_mz,
                    detectable_max_mz=detectable_max_mz,
                )

        return pl.DataFrame(
            {
                "precursor_index": out_precursor_index_arr,
                "cleavage_index": out_clevage_index_arr,
                "is_prefix": out_is_prefix_arr,
                "charge": out_charge_arr,
                "mz": out_mz_arr,
                "predicted_intensity": out_pred_intensity_arr,
                "rank": out_rank_arr,
            },
            schema=self.speclib_df_schema,
        )

    def predict_rt(
        self,
        peptide_df: pl.DataFrame,
        modification_df: pl.DataFrame,
        precursor_df: pl.DataFrame,
        batch_size: int = 512,
    ):
        precursor_ds = PeptideDataset(
            precursor_df, modification_df, peptide_df, level="peptidoform"
        )
        batch_sampler = SeqDataBatchSampler(
            precursor_ds,
            batch_grouping_column="sequence_length",
            batch_size=batch_size,
            shuffle=False,
        )
        total = batch_sampler.count_num_of_batches()
        dl = DataLoader(
            dataset=precursor_ds, batch_sampler=batch_sampler, num_workers=0
        )

        # Pre-allocate GPU buffers for zero-allocation H2D transfer
        max_token_len = peptide_df["sequence_length"].max() + 3
        mod_feat_dim = self.rt_predictor.mod_embedding.in_features
        X_aa_buf = torch.empty(
            (batch_size, max_token_len), dtype=torch.long, device=self.device
        )
        X_mod_buf = torch.empty(
            (batch_size, max_token_len, mod_feat_dim),
            dtype=torch.float32,
            device=self.device,
        )

        # Pre-allocate output arrays
        out_peptidoform_index = np.empty(modification_df.shape[0], dtype=np.uint32)
        out_ref_rt = np.empty(modification_df.shape[0], dtype=np.float32)
        offset = 0

        with torch.inference_mode():
            for batch in tqdm(
                Prefetcher(dl, transform=pin_tensor_dict),
                total=total,
                desc="Predicting RT",
                leave=True,
            ):
                peptidoform_index_arr = (
                    batch["peptidoform_index"].to(torch.uint32).numpy()
                )
                x_aa_t = batch["x_aa"]
                x_mod_t = batch["x_mod"]

                n, L = x_aa_t.shape
                X_aa_buf[:n, :L].copy_(x_aa_t, non_blocking=True)
                X_mod_buf[:n, :L, :].copy_(x_mod_t, non_blocking=True)

                y_pred = self.rt_predictor(X_aa_buf[:n, :L], X_mod_buf[:n, :L, :])

                out_peptidoform_index[offset : offset + n] = peptidoform_index_arr
                out_ref_rt[offset : offset + n] = (
                    y_pred.flatten().detach().cpu().numpy()
                )
                offset += n

        return pl.DataFrame(
            {
                "peptidoform_index": out_peptidoform_index[:offset],
                "ref_rt": out_ref_rt[:offset],
            }
        )

    def generate_spectral_lib(
        self,
        peptide_df: pl.DataFrame,
        modification_df: pl.DataFrame,
        precursor_df: pl.DataFrame,
        prefix_mass_container: PrefixMassArrayContainer,
        min_fragment_mz: float = 200,
        max_fragment_mz: float = 1800,
    ):

        rt_df = self.predict_rt(
            peptide_df=peptide_df,
            modification_df=modification_df,
            precursor_df=precursor_df,
            batch_size=512,
        )

        ms2_df = self.predict_ms2_spectra(
            peptide_df=peptide_df,
            modification_df=modification_df,
            precursor_df=precursor_df,
            prefix_mass_container=prefix_mass_container,
            batch_size=512,
            detectable_min_mz=min_fragment_mz,
            detectable_max_mz=max_fragment_mz,
        )

        modification_df = modification_df.join(
            rt_df, on="peptidoform_index", how="left"
        )

        return ms2_df, modification_df
