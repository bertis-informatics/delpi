from typing import Dict, Self, List
from collections import defaultdict
from pathlib import Path
import h5py

import polars as pl
import numpy as np

from delpi.database.peptide_database import PeptideDatabase
from delpi.search.result_manager import ResultManager, TL_DATA_GROUP
from delpi.search.config import SearchConfig
from delpi.search.dia.lfq_utils import get_apex_median_intensity


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

    def get_tl_label_df(self) -> pl.DataFrame:

        label_dfs = []
        hdf_files = self.get_hdf_files()
        for hdf_index, hdf_path in enumerate(hdf_files):
            meta = []
            with h5py.File(hdf_path, "r") as hf:
                tl_data_group = hf[TL_DATA_GROUP]
                n_tokens = list(tl_data_group)
                meta.extend(
                    (int(n), tl_data_group[n]["x_aa"].shape[0])
                    for n in n_tokens
                    if n != "config"
                )

            label_df = pl.DataFrame(
                meta,
                schema={
                    "seq_len": pl.UInt16,
                    "n_samples": pl.UInt32,
                },
                orient="row",
            ).filter(pl.col("n_samples") > 0)

            label_df = (
                label_df.with_columns(
                    pl.int_ranges(pl.col("n_samples"), dtype=pl.UInt32).alias("index")
                )
                .drop("n_samples")
                .explode("index")
                .with_columns(pl.lit(hdf_index).cast(pl.UInt32).alias("hdf_index"))
            )
            label_dfs.append(label_df)

        return pl.concat(label_dfs, how="vertical")

    def get_tl_data(self, data_keys: List[str]) -> List[Dict[str, np.ndarray]]:
        """Load all data from HDF files into memory with 3-level dict structure"""
        hdf_files = self.get_hdf_files()
        data_dicts = list()
        for hdf_index, hdf_path in enumerate(hdf_files):
            data_dict = {}
            with h5py.File(hdf_path, "r") as hf:
                tl_data_group = hf[TL_DATA_GROUP]
                n_tokens = list(tl_data_group)
                for n in n_tokens:
                    if n == "config":
                        continue
                    int_n = int(n)
                    data_dict[int_n] = {
                        key: tl_data_group[n][key][:] for key in data_keys
                    }
                    # data_dict[int_n] = {
                    #     "x_aa": tl_data_group[n]["x_aa"][:],
                    #     "x_mod": tl_data_group[n]["x_mod"][:],
                    #     "x_meta": tl_data_group[n]["x_meta"][:],
                    #     "x_intensity": tl_data_group[n]["x_intensity"][:],
                    # }
            data_dicts.append(data_dict)

        return data_dicts

    def get_run_df(self):
        run_info_dict = defaultdict(list)
        for run_index, result_manager in self._results_dict.items():
            run_info_dict["run_index"].append(run_index)
            run_info_dict["run_name"].append(result_manager.run_name)

        run_df = pl.DataFrame(
            run_info_dict,
            schema={"run_index": pl.Int32, "run_name": pl.String},
        )
        return run_df

    def save_filtered_results(
        self,
        pmsm_df: pl.DataFrame,
        src_group_key: str = "second_results",
        dst_group_key: str = "filtered_results",
    ):
        data_keys_for_quant = [
            "precursor_index",
            "frame_num",
            "frame_index",
            "search_group",
            "ms1_area",
            "quant_ab",
            "quant_theo_index",
            "quant_time_index",
        ]

        for run_index, result_manager in self._results_dict.items():
            results_dict = result_manager.read_dict(
                src_group_key,
                data_keys=data_keys_for_quant,
            )
            df = pl.DataFrame(
                {
                    "precursor_index": results_dict["precursor_index"],
                    "frame_num": results_dict["frame_num"],
                }
            ).join(
                pmsm_df.filter(pl.col("run_index") == run_index)
                .select(pl.col("precursor_index", "frame_num"))
                .with_columns(pl.lit(True).alias("present")),
                on=["precursor_index", "frame_num"],
                how="left",
            )
            mask = df["present"].is_not_null().to_numpy()
            result_manager.write_dict(
                dst_group_key, {k: v[mask] for k, v in results_dict.items()}
            )

    def get_search_results(
        self,
        group_key: str,
        load_features: bool = True,
        estimate_apex_intensity: bool = False,
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
        if estimate_apex_intensity:
            data_keys.extend(["quant_ab", "quant_time_index"])

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

                if estimate_apex_intensity:
                    quant_ab_arr = run_results_dict.pop("quant_ab")
                    quant_time_index_arr = run_results_dict.pop("quant_time_index")
                    run_results_dict["apex_median_intensity"] = (
                        get_apex_median_intensity(quant_ab_arr, quant_time_index_arr)
                    )

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
