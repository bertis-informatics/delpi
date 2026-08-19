from typing import Dict, Self, List
from collections import defaultdict
from pathlib import Path
import h5py

import polars as pl
import numpy as np

from delpi.database.peptide_database import PeptideDatabase
from delpi.search.result_manager import ResultManager
from delpi.search.config import SearchConfig
from delpi.constants import QUANT_FRAGMENTS, RT_WINDOW_LEN


class ResultsAggregator:

    def __init__(self, db_dir: Path, search_config: SearchConfig = None):
        """
        result_manager_factory: callable(h5_path:str) -> ResultManager instance
        """
        self.db_dir = db_dir
        self._results_dict: Dict[int, ResultManager] = dict()

        if search_config is not None:
            for run_index in range(len(search_config.input_files)):
                result_mgr = search_config.get_result_manager(run_index)
                self.add_result_manager(run_index, result_mgr)

    def get_result_manager(self, run_index: int) -> ResultManager:
        return self._results_dict.get(run_index)

    def add_result_manager(self, run_index: int, result_manager: ResultManager) -> Self:
        self._results_dict[run_index] = result_manager
        return self

    def get_hdf_files(self) -> List[Path]:
        return [rm.hdf_file_path for rm in self._results_dict.values()]

    def get_xic_peak_interval(self) -> float:
        xic_peak_intervals = []
        for result_manager in self._results_dict.values():
            try:
                xic_peak_interval = result_manager.read_attr("xic_peak_interval")
                xic_peak_intervals.append(xic_peak_interval)
            except KeyError:
                pass

        if len(xic_peak_intervals) == 0:
            return 2.0

        # Return the average xic_peak_interval across all runs
        return float(np.mean(xic_peak_intervals))

    def get_tl_label_df(self, tl_ms2_h5_path: Path) -> pl.DataFrame:
        """Build the transfer-learning label DataFrame from the shared
        ``transfer_learning_ms2.h5`` file (see
        :class:`~delpi.search.tl.data_prep.TransferLearningDataPreparator`).

        The file is grouped as ``/<seq_len>/<array>`` (all runs share the
        same arrays; there is no per-run grouping).
        """
        tl_ms2_h5_path = Path(tl_ms2_h5_path)
        if not tl_ms2_h5_path.exists():
            return pl.DataFrame()

        label_dfs = []
        with h5py.File(tl_ms2_h5_path, "r") as hf:
            for n in hf:
                precursor_index = hf[n]["precursor_index"][:]
                if precursor_index.shape[0] == 0:
                    continue

                n_samples = precursor_index.shape[0]
                label_dfs.append(
                    pl.DataFrame(
                        {
                            "seq_len": np.full(n_samples, int(n), dtype=np.uint16),
                            "index": np.arange(n_samples, dtype=np.uint32),
                            "precursor_index": precursor_index.astype(np.uint32, copy=False),
                        }
                    )
                )

        return pl.concat(label_dfs, how="vertical") if label_dfs else pl.DataFrame()

    def get_tl_data(
        self, tl_ms2_h5_path: Path, data_keys: List[str]
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Load all transfer-learning data from the shared HDF5 file into
        memory as a single dict keyed by sequence length (there is no
        per-run grouping)."""
        data_dict: Dict[int, Dict[str, np.ndarray]] = {}

        tl_ms2_h5_path = Path(tl_ms2_h5_path)
        if not tl_ms2_h5_path.exists():
            return data_dict

        with h5py.File(tl_ms2_h5_path, "r") as hf:
            for n in hf:
                int_n = int(n)
                data_dict[int_n] = {key: hf[n][key][:] for key in data_keys}

        return data_dict

    def get_run_df(self):
        run_info_dict = defaultdict(list)
        for run_index, result_manager in self._results_dict.items():
            run_info_dict["run_index"].append(run_index)
            run_info_dict["run_name"].append(result_manager.run_name)

        run_df = pl.DataFrame(
            run_info_dict,
            schema={"run_index": pl.UInt32, "run_name": pl.String},
        )
        return run_df

    def load_pmsm_df(
        self,
        group_key: str = "second_results",
        data_keys: List[str] = [
            "precursor_index",
            "frame_num",
            "cluster",
            "predicted_rt",
            "observed_rt",
            "logit",
        ],
    ) -> pl.DataFrame:

        assert group_key in ["first_results", "second_results"]

        results_dict = defaultdict(list)
        for run_index, result_manager in self._results_dict.items():
            run_results_dict = result_manager.read_dict(
                group_key,
                data_keys=data_keys,
            )
            n = run_results_dict["precursor_index"].shape[0]
            if n > 0:
                run_results_dict["run_index"] = np.full(n, run_index, dtype=np.uint32)
                run_results_dict["pmsm_index"] = np.arange(n, dtype=np.uint32)
                for k, v in run_results_dict.items():
                    results_dict[k].append(v)

        for k, v in results_dict.items():
            results_dict[k] = np.concatenate(v) if len(v) > 0 else v

        pmsm_df = pl.DataFrame(results_dict)
        # pmsm_df = PeptideDatabase.join(
        #     self.db_dir,
        #     pmsm_df,
        #     precursor_columns=[],
        #     modification_columns=[],
        #     peptide_columns=["is_decoy"],
        # )
        return pmsm_df

    def load_features(
        self,
        pmsm_df: pl.DataFrame,
        group_key: str = "second_results",
        feature_dim: int = 193,
    ) -> np.ndarray:

        feature_arr = np.empty((pmsm_df.shape[0], feature_dim), dtype=np.float32)

        pmsm_df = pmsm_df.with_row_index("index_")
        for run_index, result_manager in self._results_dict.items():
            sub_df = pmsm_df.filter(pl.col("run_index") == run_index)
            if sub_df.shape[0] == 0:
                continue

            # NOTE: feature_arr[sub_df["index_"]] uses advanced indexing and returns
            # a copy, so writing into it does not update feature_arr in-place.
            # Load per-run features and assign back explicitly to preserve row order.
            run_feature_arr = result_manager.load_features(
                group_key,
                feature_dim=feature_dim,
            )
            out_idx = sub_df["index_"].to_numpy()
            row_idx = sub_df["pmsm_index"].to_numpy()
            feature_arr[out_idx] = run_feature_arr[row_idx]

        return feature_arr

    def get_search_results(
        self,
        group_key: str,
        load_features: bool = True,
    ) -> tuple[pl.DataFrame, Dict[int, np.ndarray]]:

        assert group_key in ["first_results", "second_results"]

        results_dict = defaultdict(list)
        data_keys = [
            "precursor_index",
            "frame_num",
            "cluster",
            "predicted_rt",
            "observed_rt",
            "logit",
        ]

        if load_features:
            data_keys.append("features")
            data_dict = dict()
        else:
            data_dict = None

        for run_index, result_manager in self._results_dict.items():
            run_results_dict = result_manager.read_dict(
                group_key,
                data_keys=data_keys,
            )

            n = run_results_dict["precursor_index"].shape[0]
            if n > 0:
                run_results_dict["run_index"] = np.full(n, run_index, dtype=np.uint32)
                run_results_dict["pmsm_index"] = np.arange(n, dtype=np.uint32)

                if load_features:
                    features_arr = run_results_dict.pop("features")
                    data_dict[run_index] = features_arr

                for k, v in run_results_dict.items():
                    results_dict[k].append(v)

        for k, v in results_dict.items():
            results_dict[k] = np.concatenate(v) if len(v) > 0 else v

        pmsm_df = pl.DataFrame(results_dict)
        pmsm_df = PeptideDatabase.join(
            self.db_dir,
            pmsm_df,
            precursor_columns=["precursor_charge"],
            modification_columns=["mod_ids", "mod_sites"],
            peptide_columns=["peptide", "sequence_length", "is_decoy", "protein_index"],
        )

        return pmsm_df, data_dict

    def get_xic_arrays(
        self, target_pmsm_df: pl.DataFrame, group_key: str = "second_results"
    ) -> np.ndarray:
        xic_arrays = np.empty(
            (target_pmsm_df.shape[0], QUANT_FRAGMENTS, RT_WINDOW_LEN), dtype=np.float32
        )
        ms1_area_arr = np.empty(target_pmsm_df.shape[0], dtype=np.float32)

        for grp, sub_df in target_pmsm_df.with_row_index("index_").group_by(
            "run_index"
        ):
            run_index = grp[0]
            jj = sub_df["index_"]
            ii = sub_df["pmsm_index"]
            result_mgr = self.get_result_manager(run_index)

            with h5py.File(result_mgr.hdf_file_path, mode="r") as hdf_file:
                group_data = hdf_file[group_key]
                ms1_area = group_data["ms1_area"][:]
                ms1_area_arr[jj] = ms1_area[ii]
                xic_arr = group_data["xic_array"][:]
                xic_arrays[jj] = xic_arr[ii]

        return xic_arrays, ms1_area_arr
