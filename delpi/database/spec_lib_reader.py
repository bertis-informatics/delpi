from pathlib import Path

import polars as pl
import numpy as np

from delpi.model.rt_calibrator import RetentionTimeCalibrator
from delpi.database.numba.spec_lib_container import SpectralLibContainer
from delpi.database.utils import create_peptidoform_df
from delpi.chem.averagine import get_precursor_lib_df, get_isotope_indices
from delpi.lcms.base_spectra import _readonly
from delpi.utils.yaml_file import load_yaml
from delpi.constants import (
    MAX_FRAGMENTS,
    DEFAULT_NCE,
    DEFAULT_PRECURSOR_ISOTOPES,
    DEFAULT_FRAGMENT_ISOTOPES,
)
from delpi.database.spec_lib_schema import (
    SPECLIB_SCHEMA,
    fragment_offsets,
)


class SpectralLibReader:

    def __init__(
        self,
        peptide_db_path: str,
        max_fragments: int | None = None,
        max_precursor_isotopes: int = DEFAULT_PRECURSOR_ISOTOPES,
        max_fragment_isotopes: int = DEFAULT_FRAGMENT_ISOTOPES,
    ) -> None:

        self.peptide_db_path = Path(peptide_db_path)
        get_isotope_indices(max_fragment_isotopes)
        get_isotope_indices(max_precursor_isotopes)
        self.max_fragment_isotopes = max_fragment_isotopes
        self.max_precursor_isotopes = max_precursor_isotopes

        yaml_path = self.peptide_db_path / "param.yaml"
        if not yaml_path.exists():
            self.params = {}
        else:
            self.params = load_yaml(self.peptide_db_path / "param.yaml")

        if max_fragments is None:
            max_fragments = self.params.get("fragment", {}).get(
                "max_fragments", MAX_FRAGMENTS
            )
        if not isinstance(max_fragments, int) or not 1 <= max_fragments <= 255:
            raise ValueError("max_fragments must be an integer in [1, 255].")
        self.max_fragments = max_fragments

        # load precursor_df and modification_df in memory
        precursor_scan = pl.scan_parquet(self.peptide_db_path / "precursor_df.parquet")
        precursor_schema = precursor_scan.collect_schema()
        if any(
            precursor_schema.get(name) != dtype
            for name, dtype in (
                ("speclib_offset", pl.UInt32),
                ("speclib_rows", pl.UInt8),
            )
        ):
            raise ValueError(
                "Library requires speclib_offset (UInt32) and speclib_rows "
                "(UInt8) in precursor_df. Regenerate libraries in the old fixed-row format."
            )
        speclib_schema = pl.scan_parquet(
            self.peptide_db_path / "speclib_df.parquet"
        ).collect_schema()
        if speclib_schema.get("is_modloss") != pl.Boolean:
            raise ValueError(
                "Library requires an is_modloss (Boolean) column; regenerate the library."
            )
        self.all_precursor_df = precursor_scan.select(
            pl.col(
                "peptidoform_index",
                "precursor_mz",
                "precursor_charge",
                "speclib_offset",
                "speclib_rows",
            )
        ).with_row_index("precursor_index")

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

        self._peptidoform_df = None

    @property
    def library_nce(self):
        return float(self.params.get("fragment", {}).get("nce", DEFAULT_NCE))

    def calibrate_rt(self, rt_calibrator: RetentionTimeCalibrator) -> None:
        rt_df = rt_calibrator.predict(self.modification_df["ref_rt"])
        self.predicted_rt_arr[:] = rt_df["predicted_rt"].to_numpy()
        self.rt_lb_arr[:] = rt_df["rt_lb"].to_numpy()
        self.rt_ub_arr[:] = rt_df["rt_ub"].to_numpy()

    def _read_speclib_df(
        self,
        precursor_df: pl.DataFrame,
    ) -> pl.DataFrame:
        offsets = precursor_df["speclib_offset"].to_numpy()
        counts = precursor_df["speclib_rows"].to_numpy()
        if not np.array_equal(offsets[1:], offsets[:-1] + counts[:-1]):
            raise ValueError("Library precursor fragment offsets are not contiguous.")
        offset = int(offsets[0])
        num_rows = int(counts.sum())

        speclib_df = (
            pl.scan_parquet(self.peptide_db_path / "speclib_df.parquet")
            .slice(offset, num_rows)
            .select(list(SPECLIB_SCHEMA))
            .collect()
        )
        # expected_ids = np.repeat(precursor_df["precursor_index"].to_numpy(), counts)
        # if not np.array_equal(speclib_df["precursor_index"].to_numpy(), expected_ids):
        #     raise ValueError(
        #         "Library fragment rows do not match precursor offsets/counts."
        #     )
        # Recompute offsets/counts from usable rows in read_by_index_range.
        # Rank order also defines the stable cross-run quantification channels.
        return speclib_df.filter(
            (pl.col("rank") <= self.max_fragments)
            & pl.col("predicted_intensity").is_finite()
            & (pl.col("predicted_intensity") > 0)
        )

    def read_by_index_range(
        self, min_precursor_index: int, max_precursor_index: int
    ) -> SpectralLibContainer:

        if min_precursor_index is None or max_precursor_index is None:
            return None
        if min_precursor_index < 0 or max_precursor_index < min_precursor_index:
            raise ValueError("Invalid precursor index range.")

        precursor_df = self.all_precursor_df.slice(
            min_precursor_index, max_precursor_index - min_precursor_index + 1
        ).collect()
        if precursor_df.height != max_precursor_index - min_precursor_index + 1:
            raise ValueError("Precursor index range is outside the library.")
        speclib_df = self._read_speclib_df(precursor_df)
        local_ids = (
            speclib_df["precursor_index"].to_numpy().astype(np.int64)
            - min_precursor_index
        )
        counts = np.bincount(local_ids, minlength=precursor_df.height)
        if np.any(counts > self.max_fragments):
            raise ValueError(
                "Library ranks do not enforce max_fragments per precursor."
            )
        counts = counts.astype(np.uint8)
        offsets = fragment_offsets(counts)

        speclib_container = SpectralLibContainer(
            min_precursor_index=min_precursor_index,
            max_precursor_index=max_precursor_index,
            max_fragments=self.max_fragments,
            max_precursor_isotopes=self.max_precursor_isotopes,
            max_fragment_isotopes=self.max_fragment_isotopes,
            ## peptide data
            # Read-only (no copy) keeps the numpy array type stable for numba;
            # otherwise polars' readonly zero-copy views vs. copies are seen
            # as distinct types, forcing repeated njit recompilation. None of
            # the njit code writes to these fields, so no copy is needed.
            peptide_is_decoy_arr=_readonly(self.peptide_df["is_decoy"].to_numpy()),
            peptide_seq_len_arr=_readonly(
                self.peptide_df["sequence_length"].to_numpy()
            ),
            ## modification data
            mod_peptide_index_arr=_readonly(
                self.modification_df["peptide_index"].to_numpy()
            ),
            mod_ref_rt_arr=_readonly(self.modification_df["ref_rt"].to_numpy()),
            mod_predicted_rt_arr=self.predicted_rt_arr,
            mod_rt_lb_arr=self.rt_lb_arr,
            mod_rt_ub_arr=self.rt_ub_arr,
            ## precursor data
            precursor_peptidoform_index_arr=_readonly(
                precursor_df["peptidoform_index"].to_numpy()
            ),
            precursor_mz_arr=_readonly(precursor_df["precursor_mz"].to_numpy()),
            precursor_charge_arr=_readonly(precursor_df["precursor_charge"].to_numpy()),
            precursor_speclib_offset_arr=_readonly(offsets),
            precursor_speclib_rows_arr=_readonly(counts),
            ## speclib data
            speclib_cleavage_index_arr=_readonly(
                speclib_df["cleavage_index"].to_numpy()
            ),
            speclib_is_prefix_arr=_readonly(speclib_df["is_prefix"].to_numpy()),
            speclib_charge_arr=_readonly(speclib_df["charge"].to_numpy()),
            speclib_is_modloss_arr=_readonly(speclib_df["is_modloss"].to_numpy()),
            speclib_mz_arr=_readonly(speclib_df["mz"].to_numpy()),
            speclib_predicted_intensity_arr=_readonly(
                speclib_df["predicted_intensity"].to_numpy()
            ),
            ## averagine data
            averagine_min_nominal_mass=self.averagine_df.item(0, "nominal_mass"),
            averagine_max_nominal_mass=self.averagine_df.item(-1, "nominal_mass"),
            averagine_predicted_intensity=_readonly(
                self.averagine_df["predicted_intensity"].to_numpy()
            ),
            averagine_isotope_index_arr=_readonly(
                self.averagine_df["isotope_index"].to_numpy()
            ),
        )

        return speclib_container

    def _get_index_range_by_mz(
        self, min_precursor_mz: float, max_precursor_mz: float
    ) -> tuple:
        return (
            self.all_precursor_df.filter(
                pl.col("precursor_mz").is_between(min_precursor_mz, max_precursor_mz)
            )
            .select(
                min_idx=pl.col("precursor_index").min(),
                max_idx=pl.col("precursor_index").max(),
            )
            .collect()
            .row(0)
        )

    def read_by_mz_range(
        self, min_precursor_mz: float, max_precursor_mz: float
    ) -> SpectralLibContainer:

        index_range = self._get_index_range_by_mz(min_precursor_mz, max_precursor_mz)
        return self.read_by_index_range(*index_range)

    def iter_chunks_by_mz_range(
        self,
        min_precursor_mz: float,
        max_precursor_mz: float,
        max_precursors_per_chunk: int = 1_000_000,
    ):
        """Yield SpectralLibContainers in chunks for a given m/z range."""
        min_idx, max_idx = self._get_index_range_by_mz(
            min_precursor_mz, max_precursor_mz
        )
        if min_idx is None or max_idx is None:
            return
        for start in range(min_idx, max_idx + 1, max_precursors_per_chunk):
            end = min(start + max_precursors_per_chunk - 1, max_idx)
            yield self.read_by_index_range(start, end)

    def _get_index_range_by_isolation_window(
        self,
        min_isolation_mz: float,
        max_isolation_mz: float,
        isolation_lower_tol_in_da: float = 3.5,
        isolation_upper_tol_in_da: float = 1.25,
    ) -> tuple:
        min_precursor_mz = min_isolation_mz - isolation_lower_tol_in_da / pl.col(
            "precursor_charge"
        )
        max_precursor_mz = max_isolation_mz + isolation_upper_tol_in_da / pl.col(
            "precursor_charge"
        )
        return (
            self.all_precursor_df.filter(
                pl.col("precursor_mz").is_between(min_precursor_mz, max_precursor_mz)
            )
            .select(
                min_idx=pl.col("precursor_index").min(),
                max_idx=pl.col("precursor_index").max(),
            )
            .collect()
            .row(0)
        )

    def read_by_isolation_window(
        self,
        min_isolation_mz: float,
        max_isolation_mz: float,
        isolation_lower_tol_in_da: float = 3.5,
        isolation_upper_tol_in_da: float = 1.25,
    ) -> SpectralLibContainer:
        index_range = self._get_index_range_by_isolation_window(
            min_isolation_mz,
            max_isolation_mz,
            isolation_lower_tol_in_da,
            isolation_upper_tol_in_da,
        )
        return self.read_by_index_range(*index_range)

    def iter_chunks_by_isolation_window(
        self,
        min_isolation_mz: float,
        max_isolation_mz: float,
        isolation_lower_tol_in_da: float = 3.5,
        isolation_upper_tol_in_da: float = 1.25,
        max_precursors_per_chunk: int = 1_000_000,
    ):
        """Yield SpectralLibContainers in chunks for a given isolation window."""
        min_idx, max_idx = self._get_index_range_by_isolation_window(
            min_isolation_mz,
            max_isolation_mz,
            isolation_lower_tol_in_da,
            isolation_upper_tol_in_da,
        )
        if min_idx is None or max_idx is None:
            return
        for start in range(min_idx, max_idx + 1, max_precursors_per_chunk):
            end = min(start + max_precursors_per_chunk - 1, max_idx)
            yield self.read_by_index_range(start, end)

    def get_peptidoform_df(self) -> pl.DataFrame:

        if self._peptidoform_df is not None:
            return self._peptidoform_df

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
        self._peptidoform_df = create_peptidoform_df(
            peptide_df,
            modification_df,
            modified_sequence_format="delpi",
        )

        return self._peptidoform_df
