import numpy as np
import numpy as np
import polars as pl
from numba.typed import List

from delpi.chem.composition import Composition
from delpi.chem.modification_param import MOD_SEPARATOR
from delpi.constants import PROTON_MASS
from delpi.database.numba.prefix_mass_array import (
    PrefixMassArrayContainer,
    generate_prefix_mass_arrays,
)


class PrecursorGenerator:

    def __init__(
        self,
        min_charge: int = 1,
        max_charge: int = 4,
        min_mz: float = 300,
        max_mz: float = 1800,
    ):
        self.min_charge = min_charge
        self.max_charge = max_charge
        self.min_mz = min_mz
        self.max_mz = max_mz

    @property
    def param_dict(self):
        return {
            "min_charge": self.min_charge,
            "max_charge": self.max_charge,
            "min_mz": self.min_mz,
            "max_mz": self.max_mz,
        }

    def _generate_prefix_mass_array(self, peptide_df, modification_df):

        peptide_df = peptide_df
        mod_df = modification_df

        # peptides = peptide_df["peptide"].to_numpy().astype(str)
        peptide_list = List(peptide_df["peptide"])

        ## a bit messy codes for faster computation
        count_mods = (
            pl.when(pl.col("mod_ids").is_null())
            .then(0)
            .otherwise(pl.col("mod_ids").str.split(MOD_SEPARATOR).list.len())
        )

        # index_arr consitsts of [peptide_index, start_index, stop_index]
        # mod_ids and mod_sites can be obtained from mod_info_arr
        # (start_index, stop_index) are row indexes of mod_info_arr
        index_arr = (
            mod_df.with_columns(count_mods.cum_sum().alias("stop_index"))
            .select(
                pl.col("peptide_index"),
                pl.col("stop_index").shift(1, fill_value=0).alias("start_index"),
                pl.col("stop_index"),
            )
            .to_numpy()
        )

        # mod_info_arr consists of (mod_ids, mod_sites)
        mod_info_arr = (
            mod_df.filter(pl.col("mod_ids").is_not_null())
            .select(
                pl.col("mod_ids").str.split(MOD_SEPARATOR).cast(pl.List(pl.UInt32)),
                pl.col("mod_sites").str.split(MOD_SEPARATOR).cast(pl.List(pl.Int8)),
            )
            .explode(pl.all())
            .to_numpy()
        )

        flatten_mass_array, stop_row_indices = generate_prefix_mass_arrays(
            peptide_list, index_arr, mod_info_arr
        )

        return flatten_mass_array, stop_row_indices

    def generate_precursors(
        self, peptide_df: pl.DataFrame, modification_df: pl.DataFrame
    ):

        flatten_mass_array, stop_row_indices = self._generate_prefix_mass_array(
            peptide_df, modification_df
        )

        # Apply per-peptide fragment mass shifts for diann-style decoys before
        # precursor mass is read from the array.
        if "decoy_n_fragment_shift" in peptide_df.columns:
            self._apply_diann_shifts(
                flatten_mass_array, stop_row_indices, modification_df, peptide_df
            )

        # Apply per-peptide fragment mass shifts for diann-style decoys.
        # This must happen before precursor_mass is read from the array.
        if "decoy_n_fragment_shift" in peptide_df.columns:
            self._apply_diann_shifts(
                flatten_mass_array, stop_row_indices, modification_df, peptide_df
            )

        precursor_mass = flatten_mass_array[stop_row_indices - 1] + Composition.H2O.mass
        modification_df = modification_df.with_columns(
            # mass_array_stop_index=stop_row_indices,
            precursor_mass=precursor_mass
        )

        get_mz = (
            (pl.col("precursor_mass") / pl.col("precursor_charge")) + PROTON_MASS
        ).alias("precursor_mz")
        get_charge = pl.int_ranges(
            self.min_charge, self.max_charge + 1, dtype=pl.UInt8
        ).alias("precursor_charge")

        precursor_df = (
            modification_df.select(
                pl.col("peptidoform_index", "precursor_mass"), get_charge
            )
            .explode("precursor_charge")
            .select(pl.col("peptidoform_index", "precursor_charge"), get_mz)
            .filter(pl.col("precursor_mz").is_between(self.min_mz, self.max_mz))
            .sort(pl.col("precursor_mz", "peptidoform_index"))
        )

        # precursor_df = precursor_df.with_row_index("precursor_index")
        prefix_mass_container = PrefixMassArrayContainer(
            prefix_mass_array=flatten_mass_array, mass_array_stop_index=stop_row_indices
        )
        # peptidoform_prefix_mass_array = flatten_mass_array
        # peptidoform_mass_array_stop_index = stop_row_indices

        return (precursor_df, modification_df, prefix_mass_container)

    @staticmethod
    def _apply_diann_shifts(
        flatten_mass_array: np.ndarray,
        stop_row_indices: np.ndarray,
        modification_df: pl.DataFrame,
        peptide_df: pl.DataFrame,
    ) -> None:
        """Apply N- and C-side fragment mass shifts for diann decoys in-place.

        For each peptidoform that carries non-zero shifts (i.e. diann decoys):

        - ``decoy_n_fragment_shift`` (delta for second residue) is added to
          prefix-mass positions 1 .. n-1.  This shifts b2 and all higher b-ions,
          and leaves shared-residue y-ion differences unchanged.
        - ``decoy_c_fragment_shift`` (delta for second-to-last residue) is added
          to prefix-mass positions n-2 .. n-1.

        Both shifts are applied before ``precursor_mass`` is read from the array,
        so the precursor m/z is updated consistently.
        """
        shift_df = (
            modification_df.select("peptide_index")
            .join(
                peptide_df.select(
                    "peptide_index",
                    "decoy_n_fragment_shift",
                    "decoy_c_fragment_shift",
                ),
                on="peptide_index",
                how="left",
            )
        )
        n_shifts = shift_df["decoy_n_fragment_shift"].fill_null(0.0).to_numpy()
        c_shifts = shift_df["decoy_c_fragment_shift"].fill_null(0.0).to_numpy()

        for i in np.where((n_shifts != 0.0) | (c_shifts != 0.0))[0]:
            start = int(stop_row_indices[i - 1]) if i > 0 else 0
            stop  = int(stop_row_indices[i])
            n     = stop - start          # number of residues in this peptidoform
            delta_n = float(n_shifts[i])
            delta_c = float(c_shifts[i])
            # N-side: second residue is at cumsum index 1; delta propagates to end.
            if delta_n != 0.0 and n > 1:
                flatten_mass_array[start + 1 : stop] += delta_n
            # C-side: second-to-last residue is at cumsum index n-2.
            if delta_c != 0.0 and n >= 2:
                flatten_mass_array[stop - 2 : stop] += delta_c
