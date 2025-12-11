from pathlib import Path

import polars as pl
import numpy as np

from delpi.model.rt_calibrator import RetentionTimeCalibrator
from delpi.database.numba.spec_lib_container import SpectralLibContainer
from delpi.database.utils import create_peptidoform_df
from delpi.chem.averagine import get_precursor_lib_df
from delpi.utils.yaml_file import load_yaml
from delpi.constants import MAX_FRAGMENTS


class SpectralLibReader:

    def __init__(
        self,
        peptide_db_path: str,
        max_fragments: int = MAX_FRAGMENTS,
        max_precursor_isotopes: int = 3,
        max_fragment_isotopes: int = 2,
    ) -> None:

        self.peptide_db_path = Path(peptide_db_path)
        self.max_fragments = max_fragments
        self.max_fragment_isotopes = max_fragment_isotopes
        self.max_precursor_isotopes = max_precursor_isotopes

        yaml_path = self.peptide_db_path / "param.yaml"
        if not yaml_path.exists():
            self.params = {}
        else:
            self.params = load_yaml(self.peptide_db_path / "param.yaml")

        # load precursor_df and modification_df in memory
        self.all_precursor_df = (
            pl.scan_parquet(self.peptide_db_path / "precursor_df.parquet")
            .select(pl.col("peptidoform_index", "precursor_mz", "precursor_charge"))
            .with_row_index("precursor_index")
        )

        self.peptide_df = (
            pl.scan_parquet(self.peptide_db_path / "peptide_df.parquet")
            .select(pl.col("is_decoy"), pl.col("sequence_length").cast(pl.UInt8))
            .collect()
        )
        self.modification_df = (
            pl.scan_parquet(self.peptide_db_path / "modification_df.parquet")
            .select(pl.col("peptide_index", "ref_rt"))
            .collect()
        )

        n = self.modification_df.shape[0]
        self.rt_lb_arr = np.full(n, np.nan, dtype=np.float32)
        self.rt_ub_arr = np.full(n, np.nan, dtype=np.float32)
        self.predicted_rt_arr = np.full(n, np.nan, dtype=np.float32)

        self.averagine_df = get_precursor_lib_df(max_isotopes=max_precursor_isotopes)

    def calibrate_rt(self, rt_calibrator: RetentionTimeCalibrator) -> None:
        rt_df = rt_calibrator.predict(self.modification_df["ref_rt"])
        self.predicted_rt_arr[:] = rt_df["predicted_rt"].to_numpy()
        self.rt_lb_arr[:] = rt_df["rt_lb"].to_numpy()
        self.rt_ub_arr[:] = rt_df["rt_ub"].to_numpy()

    def _read_speclib_df(
        self,
        min_precursor_index: int,
        max_precursor_index: int,
    ) -> pl.DataFrame:

        lib_max_fragments = MAX_FRAGMENTS
        assert self.max_fragments <= lib_max_fragments

        num_precursors = max_precursor_index - min_precursor_index + 1
        offset = min_precursor_index * lib_max_fragments
        num_rows = num_precursors * lib_max_fragments

        speclib_df = (
            pl.scan_parquet(self.peptide_db_path / "speclib_df.parquet")
            .slice(offset, num_rows)
            .select(
                pl.col(
                    "precursor_index",
                    "cleavage_index",
                    "is_prefix",
                    "charge",
                    "mz",
                    "predicted_intensity",
                    "rank",
                )
            )
        )

        if self.max_fragments < lib_max_fragments:
            speclib_df = speclib_df.filter(pl.col("rank") <= self.max_fragments)

        return speclib_df.collect()

    def read_by_index_range(
        self, min_precursor_index: int, max_precursor_index: int
    ) -> SpectralLibContainer:

        if min_precursor_index is None or max_precursor_index is None:
            return None

        precursor_df = self.all_precursor_df.slice(
            min_precursor_index, max_precursor_index - min_precursor_index + 1
        ).collect()
        speclib_df = self._read_speclib_df(min_precursor_index, max_precursor_index)

        speclib_container = SpectralLibContainer(
            min_precursor_index=min_precursor_index,
            max_precursor_index=max_precursor_index,
            max_fragments=self.max_fragments,
            max_precursor_isotopes=self.max_precursor_isotopes,
            max_fragment_isotopes=self.max_fragment_isotopes,
            ## peptide data
            peptide_is_decoy_arr=self.peptide_df["is_decoy"].to_numpy(),
            peptide_seq_len_arr=self.peptide_df["sequence_length"].to_numpy(),
            ## modification data
            mod_peptide_index_arr=self.modification_df["peptide_index"].to_numpy(),
            mod_ref_rt_arr=self.modification_df["ref_rt"].to_numpy(),
            mod_predicted_rt_arr=self.predicted_rt_arr,
            mod_rt_lb_arr=self.rt_lb_arr,
            mod_rt_ub_arr=self.rt_ub_arr,
            ## precursor data
            precursor_peptidoform_index_arr=precursor_df[
                "peptidoform_index"
            ].to_numpy(),
            precursor_mz_arr=precursor_df["precursor_mz"].to_numpy(),
            precursor_charge_arr=precursor_df["precursor_charge"].to_numpy(),
            ## speclib data
            speclib_cleavage_index_arr=speclib_df["cleavage_index"].to_numpy(),
            speclib_is_prefix_arr=speclib_df["is_prefix"].to_numpy(),
            speclib_charge_arr=speclib_df["charge"].to_numpy(),
            speclib_mz_arr=speclib_df["mz"].to_numpy(),
            speclib_predicted_intensity_arr=speclib_df[
                "predicted_intensity"
            ].to_numpy(),
            ## averagine data
            averagine_min_nominal_mass=self.averagine_df.item(0, "nominal_mass"),
            averagine_max_nominal_mass=self.averagine_df.item(-1, "nominal_mass"),
            averagine_predicted_intensity=self.averagine_df[
                "predicted_intensity"
            ].to_numpy(),
            averagine_isotope_index_arr=self.averagine_df["isotope_index"].to_numpy(),
        )

        return speclib_container

    def read_by_mz_range(
        self, min_precursor_mz: float, max_precursor_mz: float
    ) -> SpectralLibContainer:

        index_range = (
            self.all_precursor_df.filter(
                pl.col("precursor_mz").is_between(min_precursor_mz, max_precursor_mz)
            )
            .select(
                min_mz=pl.col("precursor_index").min(),
                max_mz=pl.col("precursor_index").max(),
            )
            .collect()
            .row(0)
        )

        return self.read_by_index_range(*index_range)

    def read_by_isolation_window(
        self,
        min_isolation_mz: float,
        max_isolation_mz: float,
        isolation_lower_tol_in_da: float = 3.5,
        isolation_upper_tol_in_da: float = 1.25,
    ) -> SpectralLibContainer:
        min_precursor_mz = min_isolation_mz - isolation_lower_tol_in_da / pl.col(
            "precursor_charge"
        )
        max_precursor_mz = max_isolation_mz + isolation_upper_tol_in_da / pl.col(
            "precursor_charge"
        )
        index_range = (
            self.all_precursor_df.filter(
                pl.col("precursor_mz").is_between(min_precursor_mz, max_precursor_mz)
            )
            .select(
                min_mz=pl.col("precursor_index").min(),
                max_mz=pl.col("precursor_index").max(),
            )
            .collect()
            .row(0)
        )
        return self.read_by_index_range(*index_range)

    def get_peptidoform_df(self) -> pl.DataFrame:

        peptide_df = (
            pl.scan_parquet(self.peptide_db_path / "peptide_df.parquet")
            .select(
                pl.col("peptide", "is_decoy"), pl.col("sequence_length").cast(pl.UInt8)
            )
            .collect()
        )
        modification_df = (
            pl.scan_parquet(self.peptide_db_path / "modification_df.parquet")
            .select(
                pl.col(
                    "peptidoform_index",
                    "peptide_index",
                    "mod_ids",
                    "mod_sites",
                    "ref_rt",
                )
            )
            .collect()
        )
        peptidoform_df = create_peptidoform_df(
            peptide_df,
            modification_df,
            modified_sequence_format="delpi",
        )

        return peptidoform_df
