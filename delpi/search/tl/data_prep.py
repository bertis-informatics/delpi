from pathlib import Path
from typing import Dict, Union
from collections import defaultdict
from tqdm import tqdm

import h5py
import polars as pl
import numpy as np

from pymsio import MassSpecData, ReaderFactory
from delpi.lcms.fragmentation import Fragmentation
from delpi.lcms.neutral_loss import NeutralLoss
from delpi.database.precursor_generator import PrecursorGenerator
from delpi.database.numba.prefix_mass_array import PrefixMassArrayContainer
from delpi.search.tl.numba import get_intensity_arr, get_mz_arr
from delpi.model.spec_lib.aa_encoder import (
    encode_sequence,
    encode_modification,
    MAX_MOD_COUNT,
)


class TransferLearningDataPreparator:
    """MS2 intensity / RT data preparation for transfer learning.

    Extracts fragment-ion intensities and RT of identified PmSMs from a raw
    LC-MS/MS file and appends them to one HDF5 file shared by all runs
    (``/{sequence_length}/{array_name}``), not to each run's result file.
    """

    def __init__(
        self,
        nce: float = 30.0,
        frag_method: int = 0,  # 0: HCD, 1: CID
        mass_analyzer: int = 0,  # 0: FTMS, 1: ITMS
        tolerance_in_ppm: float = 10.0,
        apply_phospho: bool = False,
        min_charge: int = 1,
        max_charge: int = 2,
    ):
        """
        Args:
            nce: Normalized collision energy stored in ``x_meta``.
            apply_phospho: Also match phospho neutral-loss fragments.
            min_charge, max_charge: Fragment charge range.
        """
        self.nce = nce
        self.frag_method = frag_method
        self.mass_analyzer = mass_analyzer
        self.tolerance_in_ppm = tolerance_in_ppm
        self.apply_phospho = apply_phospho
        self.min_charge = min_charge
        self.max_charge = max_charge

        self.precursor_gen = PrecursorGenerator()
        self._setup_fragmentation()

    def _setup_fragmentation(self):
        """Initialize fragmentation configuration"""
        neutral_losses = [NeutralLoss.NO_LOSS]
        if self.apply_phospho:
            neutral_losses.append(NeutralLoss.H3O4P)

        self.fragmentation = Fragmentation(
            min_charge=self.min_charge,
            max_charge=self.max_charge,
            max_fragment_isotopes=1,
            neutral_losses=neutral_losses,
        )

        self.ion_type_container = self.fragmentation.get_ion_types()

    def prepare_training_data(
        self,
        pmsm_df: pl.DataFrame,
        lcms_data: MassSpecData,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Training arrays of the PmSMs in `pmsm_df`, as
        ``{sequence_length: {array_name: np.ndarray}}``."""

        self._validate_dataframe(pmsm_df)

        # TODO: avoid learning several peptidoforms from one spectrum (shared
        # fragment peaks), e.g. keep the best-scoring PmSM per frame
        # tl_pmsm_df = pmsm_df.group_by(["frame_num"]).agg(
        #     pl.all().sort_by("score").last()
        # )

        collected_data = self._collect_ms2_intensity_data(pmsm_df, lcms_data)

        return collected_data

    def extract_and_save(
        self,
        pmsm_df: Union[pl.DataFrame, Path, str],
        raw_file_path: Union[Path, str],
        h5_file_path: Union[Path, str],
    ) -> None:
        """Extract one run's MS2/RT training data and append it to a shared
        HDF5 file (see :meth:`save_to_hdf`).

        Parameters
        ----------
        pmsm_df:
            Pre-filtered PmSMs of this run, or a parquet path. Required
            columns: ``precursor_index``, ``frame_num``, ``peptide``,
            ``mod_ids``, ``mod_sites``, ``precursor_charge``,
            ``sequence_length``.
        raw_file_path:
            Raw LC-MS/MS file of this run (``.raw``, ``.mzml`` or ``.mzml.gz``).
        """
        raw_file_path = Path(raw_file_path)
        h5_file_path = Path(h5_file_path)
        h5_file_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(pmsm_df, (str, Path)):
            pmsm_df = pl.read_parquet(pmsm_df)

        if len(pmsm_df) == 0:
            return

        reader = ReaderFactory.get_reader(raw_file_path)
        lcms_data = reader.load()

        collected_data = self.prepare_training_data(pmsm_df, lcms_data)
        self.save_to_hdf(collected_data, h5_file_path)

    @staticmethod
    def save_to_hdf(
        collected_data: Dict[int, Dict[str, np.ndarray]],
        h5_file_path: Union[Path, str],
    ) -> None:
        """Append `collected_data` to a shared HDF5 file (created if missing).

        Layout: ``/<seq_len>/<array_name>``, LZF-compressed with an unlimited
        first axis; all runs append to the same arrays (no per-run group).

        Parameters
        ----------
        collected_data:
            ``{sequence_length: {array_name: np.ndarray}}`` from
            :meth:`prepare_training_data`.
        """
        h5_file_path = Path(h5_file_path)
        h5_file_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(h5_file_path, "a") as hf:
            for seq_len, data_dict in collected_data.items():
                if not data_dict:
                    continue
                grp = hf.require_group(str(seq_len))

                for array_name, array_data in data_dict.items():
                    if array_data.ndim == 1:
                        chunk_shape = (1,)
                        max_shape = (None,)
                    else:
                        chunk_shape = (1, *array_data.shape[1:])
                        max_shape = (None, *array_data.shape[1:])

                    if array_name not in grp:
                        grp.create_dataset(
                            array_name,
                            data=array_data,
                            compression="lzf",
                            chunks=chunk_shape,
                            maxshape=max_shape,
                        )
                    else:
                        ds = grp[array_name]
                        n_existing = ds.shape[0]
                        n_new = array_data.shape[0]
                        ds.resize(n_existing + n_new, axis=0)
                        ds[-n_new:] = array_data

    def _validate_dataframe(self, df: pl.DataFrame):
        """Validate required columns in input dataframe"""
        required_columns = [
            "precursor_index",
            "frame_num",
            "peptide",
            "mod_ids",
            "mod_sites",
            "precursor_charge",
            "sequence_length",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _collect_ms2_intensity_data(
        self, pmsm_df: pl.DataFrame, lcms_data
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Collect MS2 intensity data from PMSM dataframe"""

        frame_num_to_rt_arr = np.empty(
            lcms_data.meta_df.item(-1, "frame_num") + 1, dtype=np.float32
        )
        frame_num_to_rt_arr[lcms_data.meta_df["frame_num"]] = lcms_data.meta_df[
            "time_in_seconds"
        ]

        collected_data = dict()

        for seq_len in pmsm_df["sequence_length"].unique():
            collected_data[seq_len] = dict()

        df = pmsm_df.select(
            pl.col(
                "precursor_index",
                "frame_num",
                "peptide",
                "precursor_charge",
                "sequence_length",
                "mod_ids",
                "mod_sites",
            )
        )

        for seq_len, sub_df in tqdm(
            df.group_by(["sequence_length"]), total=len(collected_data), desc="TL-Prep"
        ):
            seq_len = seq_len[0]
            batch_data = self._process_sequence_length_batch(
                sub_df, seq_len, lcms_data, frame_num_to_rt_arr
            )
            for key, value in batch_data.items():
                collected_data[seq_len][key] = value

        return collected_data

    def _process_sequence_length_batch(
        self,
        sub_df: pl.DataFrame,
        seq_len: int,
        lcms_data: MassSpecData,
        frame_num_to_rt_arr: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Process batch of specific sequence length"""
        n = sub_df.shape[0]
        num_cleavages = seq_len - 1
        sub_df = sub_df.sort(["frame_num"]).with_row_index("peptide_index")

        flatten_mass_array, stop_row_indices = (
            self.precursor_gen._generate_prefix_mass_array(
                peptide_df=sub_df.select(pl.col("peptide_index", "peptide")),
                modification_df=sub_df.select(
                    pl.col("peptide_index", "mod_ids", "mod_sites")
                ),
            )
        )
        prefix_mass_container = PrefixMassArrayContainer(
            prefix_mass_array=flatten_mass_array, mass_array_stop_index=stop_row_indices
        )

        x_aa = np.empty((n, seq_len + 2), dtype=np.int8)
        x_mod = np.full((n, MAX_MOD_COUNT, 2), -1, dtype=np.int16)
        x_meta = np.empty((n, 4), dtype=np.float32)
        x_intensity = np.empty(
            (n, num_cleavages, self.ion_type_container.charge_arr.shape[0]),
            dtype=np.float32,
        )
        x_rt = np.empty((n,), dtype=np.float32)

        # x_meta columns: precursor charge, nce, frag_method, mass_analyzer
        x_meta[:, 1] = self.nce
        x_meta[:, 2] = self.frag_method
        x_meta[:, 3] = self.mass_analyzer

        prev_frame_num = -1
        prev_peak_arr = None
        rt_in_seconds = 0.0
        for i, (frame_num, peptide, mod_ids, mod_sites, precursor_charge) in enumerate(
            sub_df.select(
                pl.col(
                    "frame_num", "peptide", "mod_ids", "mod_sites", "precursor_charge"
                )
            ).iter_rows()
        ):
            _ = encode_sequence(peptide, out_arr=x_aa[i])
            _ = encode_modification(mod_sites, mod_ids, out_arr=x_mod[i])
            x_meta[i, 0] = precursor_charge

            prefix_mass_arr = prefix_mass_container.get_prefix_mass_array(i)
            if prev_frame_num == frame_num:
                peak_arr = prev_peak_arr
            else:
                peak_arr = lcms_data.get_frame(frame_num=frame_num)
                rt_in_seconds = frame_num_to_rt_arr[frame_num]

            mz_arr = get_mz_arr(prefix_mass_arr, self.ion_type_container)
            x_intensity[i] = get_intensity_arr(
                mz_arr, peak_arr.mz, peak_arr.ab, self.tolerance_in_ppm
            )
            x_rt[i] = rt_in_seconds

            prev_frame_num = frame_num
            prev_peak_arr = peak_arr

        return {
            "precursor_index": sub_df["precursor_index"].to_numpy(),
            "x_intensity": x_intensity,
            "x_aa": x_aa,
            "x_mod": x_mod,
            "x_meta": x_meta,
            "x_rt": x_rt,
        }

    def get_data_statistics(
        self, collected_data: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict:
        """Return data statistics"""
        stats = {
            "sequence_lengths": list(collected_data.keys()),
            "total_samples": 0,
            "samples_per_length": {},
            "charge_distribution": defaultdict(int),
            "nce_distribution": defaultdict(int),
        }

        for seq_len, data_dict in collected_data.items():
            if not data_dict:
                continue

            n_samples = len(data_dict["x_aa"])
            stats["samples_per_length"][seq_len] = n_samples
            stats["total_samples"] += n_samples

            charges = data_dict["x_meta"][:, 0]
            for charge in charges:
                stats["charge_distribution"][int(charge)] += 1

            nces = data_dict["x_meta"][:, 1]
            for nce in nces:
                stats["nce_distribution"][float(nce)] += 1

        return stats

