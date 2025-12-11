from typing import Union
from pathlib import Path

import polars as pl
import numpy as np

import torch

from delpi.lcms.base_ion_type import BaseIonType
from delpi.model.spec_lib import Ms2SpectrumPredictor

from delpi.database.peptide_database import PeptideDatabase
from delpi.database.spec_lib_generator import SpectralLibGenerator
from delpi.database.precursor_generator import PrecursorGenerator
from delpi.model.spec_lib.rt_predictor import RetentionTimePredictor


class RefinedSpectralLibGenerator(SpectralLibGenerator):

    def __init__(
        self,
        rt_predictor: RetentionTimePredictor,
        ms2_predictor: Ms2SpectrumPredictor,
        apply_phospho,
        min_charge: int = 1,
        max_charge: int = 2,
        prefix_ion_type=BaseIonType.B,
        suffix_ion_type=BaseIonType.Y,
        max_fragments=16,
        device: Union[str, torch.device] = "cuda:0",
        *args,
        **kwargs,
    ):
        super().__init__(
            rt_predictor=rt_predictor,
            ms2_predictor=ms2_predictor,
            min_charge=min_charge,
            max_charge=max_charge,
            prefix_ion_type=prefix_ion_type,
            suffix_ion_type=suffix_ion_type,
            max_fragments=max_fragments,
            apply_phospho=apply_phospho,
            device=device,
            *args,
            **kwargs,
        )

        self.peptide_df = None
        self.modification_df = None
        self.precursor_df = None
        self.prefix_mass_container = None
        self.speclib_df = None

    def _build_database(self, db_dir: Path, precursor_index_arr: np.ndarray):

        pmsm_df = PeptideDatabase.join(
            db_dir,
            pl.DataFrame({"precursor_index": precursor_index_arr}),
            precursor_columns=["precursor_charge"],
            modification_columns=["mod_ids", "mod_sites", "ref_rt"],
            peptide_columns=["peptide", "is_decoy", "sequence_length", "protein_index"],
        ).rename(
            {
                "precursor_index": "g_precursor_index",
                "peptide_index": "g_peptide_index",
                "peptidoform_index": "g_peptidoform_index",
            }
        )

        peptide_df = (
            pmsm_df.unique(subset="g_peptide_index", keep="first")
            .select(
                pl.col(
                    "g_peptide_index",
                    "peptide",
                    "is_decoy",
                    "sequence_length",
                    "protein_index",
                )
            )
            .with_row_index("peptide_index")
        )

        modification_df = (
            pmsm_df.unique(subset="g_peptidoform_index", keep="first")
            .select(
                pl.col(
                    "g_peptide_index",
                    "g_peptidoform_index",
                    "mod_ids",
                    "mod_sites",
                    "ref_rt",
                )
            )
            .with_row_index("peptidoform_index")
            .join(
                peptide_df.select(pl.col("g_peptide_index", "peptide_index")),
                on="g_peptide_index",
                how="left",
            )
        )

        precursor_gen = PrecursorGenerator()
        precursor_df, modification_df, prefix_mass_container = (
            precursor_gen.generate_precursors(peptide_df, modification_df)
        )
        precursor_df = (
            precursor_df.join(
                modification_df.select(
                    pl.col("peptidoform_index", "g_peptidoform_index")
                ),
                on="peptidoform_index",
                how="left",
            )
            .join(
                pmsm_df.select(
                    pl.col(
                        "g_peptidoform_index", "precursor_charge", "g_precursor_index"
                    )
                ),
                on=["g_peptidoform_index", "precursor_charge"],
                how="inner",
            )
            .sort("precursor_mz")
            .with_row_index("precursor_index")
        )

        self.peptide_df = peptide_df
        self.modification_df = modification_df
        self.precursor_df = precursor_df
        self.prefix_mass_container = prefix_mass_container

    def generate_spectral_lib(
        self,
        db_dir: Path,
        precursor_index_arr: np.ndarray,
        min_fragment_mz: float = 200,
        max_fragment_mz: float = 1800,
    ):

        self._build_database(db_dir, precursor_index_arr)
        rt_df = self.predict_rt(
            peptide_df=self.peptide_df,
            modification_df=self.modification_df,
            precursor_df=self.precursor_df,
            batch_size=512,
        )

        ms2_df = self.predict_ms2_spectra(
            peptide_df=self.peptide_df,
            modification_df=self.modification_df,
            precursor_df=self.precursor_df,
            prefix_mass_container=self.prefix_mass_container,
            batch_size=512,
            detectable_min_mz=min_fragment_mz,
            detectable_max_mz=max_fragment_mz,
        )
        self.speclib_df = ms2_df
        self.modification_df = self.modification_df.select(pl.exclude("ref_rt")).join(
            rt_df, on="peptidoform_index", how="left"
        )

    def save(self, save_dir: Path):
        """Save spectral library to parquet files."""
        save_dir.mkdir(parents=True, exist_ok=True)
        self.peptide_df.write_parquet(save_dir / "peptide_df.parquet")
        self.modification_df.write_parquet(save_dir / "modification_df.parquet")
        self.precursor_df.write_parquet(save_dir / "precursor_df.parquet")
        self.speclib_df.write_parquet(save_dir / "speclib_df.parquet")


def test():

    from delpi.search.result_manager import ResultManager
    from delpi.model.spec_lib.ms2_predictor import Ms2SpectrumPredictor
    import h5py

    # hf = h5py.File(
    #     r"/data1/MassSpecData/DIA_LIBD/delpi/20240215_Ast_Neo_150uID_4mz_DIA_400-900_10maxIT_250agc_27nce-24m_Control_C1.delpi.h5"
    # )
    # hf.close()

    result_mgr = ResultManager(
        db_path=r"/data1/MassSpecData/DIA_LIBD/delpi/spec",
        hdf_file_path=r"/data1/MassSpecData/DIA_LIBD/delpi/20240215_Ast_Neo_150uID_4mz_DIA_400-900_10maxIT_250agc_27nce-24m_Control_C1.delpi.h5",
        run_name="20240215_Ast_Neo_150uID_4mz_DIA_400-900_10maxIT_250agc_27nce-24m_Control_C1",
    )
    pmsm_df = result_mgr.load_search_results(
        data_keys=["precursor_index"], with_features=False
    )

    model = Ms2SpectrumPredictor.load_from_checkpoint(
        r"/data1/MassSpecData/DIA_LIBD/delpi/lightning_logs/ms2_predictor_tl/checkpoints/epoch=9.ckpt"
    )

    apply_phospho = False

    spec_generator = RefinedSpectralLibGenerator(
        apply_phospho=apply_phospho, ms2_predictor=model
    )
    # self = spec_generator
    spec_generator.generate_spectral_lib(pmsm_df)

    ms2_df = spec_generator.spec_lib_df
    ms2_df = ms2_df.join(
        spec_generator.precursor_df.select(
            pl.col("precursor_index", "g_precursor_index")
        ),
        on="precursor_index",
        how="left",
    )

    df = pl.read_parquet(result_mgr.db_path / "speclib_df.parquet")

    ms2_df.filter(pl.col("g_precursor_index") == 8164477)
    df.filter(pl.col("precursor_index") == 8164477)

    spec_generator.peptide_df
    spec_generator.modification_df
    spec_generator.precursor_df
    pl.scan_parquet(result_mgr.db_path / "precursor_df.parquet").head().collect()

    # protein_index in peptide_df
    # ref_rt in modification_df
