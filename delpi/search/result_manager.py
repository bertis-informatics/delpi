"""
Result Manager for DelPi

This module provides the ResultManager class for reading and writing
search results in HDF5 format with database integration.
"""

from pathlib import Path
from typing import Union, List, Dict
import logging

import h5py
import numpy as np
import polars as pl
import pandas as pd


TL_DATA_GROUP = "tl_data"


logger = logging.getLogger(__name__)


class ResultManager:
    """
    Manages reading and writing of search results in HDF5 format.

    This class provides a unified interface for:
    - Loading search results with database joins
    - Writing search results to HDF5
    - Managing features and metadata
    """

    def __init__(self, run_name: str, output_dir: Path):
        """
        Initialize ResultManager.
        """
        self.run_name = run_name
        self.hdf_file_path = self.get_hdf_file_path(output_dir, run_name)
        self.hdf_file_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self.hdf_file_path.parent

    @staticmethod
    def get_hdf_file_path(output_dir: Path, run_name: str) -> Path:
        return output_dir / f"{run_name}.delpi.h5"

    def write_attr(self, attr_name: str, attr_value: Union[str, int, float]) -> None:
        with h5py.File(self.hdf_file_path, "a") as f:
            f.attrs[attr_name] = attr_value

    def read_attr(self, attr_name: str) -> Union[str, int, float]:
        with h5py.File(self.hdf_file_path, "r") as f:
            if attr_name in f.attrs:
                return f.attrs[attr_name]
            else:
                raise KeyError(f"Attribute `{attr_name}` not found in HDF5 file.")

    def write_df(
        self,
        df: pl.DataFrame,
        key: str,
        complib: str = "blosc:zstd",
        complevel: int = 4,
    ) -> None:
        df.to_pandas().to_hdf(
            self.hdf_file_path,
            key=key,
            mode="a",
            format="fixed",
            complib=complib,
            complevel=complevel,
        )

    def read_df(self, key: str) -> pl.DataFrame:
        return pl.from_pandas(pd.read_hdf(self.hdf_file_path, key=key))

    def read_dict(self, group_key: str, data_keys: List[str]) -> Dict[str, np.ndarray]:
        """
        Load search results from HDF5 file.

        Args:
            data_keys: List of data keys to load
            hdf_path: Path to HDF5 file (uses instance path if None)
            mode: File opening mode (uses instance mode if None)
        Returns:
            Dictionary of loaded data arrays
        """

        with h5py.File(self.hdf_file_path, mode="r") as hdf_file:
            if group_key not in hdf_file:
                raise KeyError(f"Group `{group_key}` not found in HDF5 file.")

            result_group = hdf_file[group_key]

            results = {}
            for key in data_keys:
                if key in result_group:
                    results[key] = result_group[key][...]
                # else:
                #     raise KeyError(f"Dataset `{key}` not found in group `{group_key}`.")

        return results

    def write_dict(
        self,
        group_key: str,
        data_dict: dict,
        chunk_size: int = 512,
    ) -> None:
        """
        Write search results to HDF5 file.
        """

        with h5py.File(self.hdf_file_path, mode="a") as hdf_file:

            if group_key in hdf_file:
                result_group = hdf_file[group_key]
            else:
                result_group = hdf_file.create_group(group_key)

            # Write main data arrays
            for key, data in data_dict.items():
                # [NOTE] feature arrays are chunked by 1 row for quick random access during training
                chunk_row_size = 1 if key == "features" else chunk_size
                if isinstance(data, np.ndarray):
                    self._write_data(result_group, key, data, chunk_row_size)
                elif hasattr(data, "__array__"):  # Handle other array-like objects
                    self._write_data(
                        result_group, key, np.asarray(data), chunk_row_size
                    )
                else:
                    raise TypeError(
                        f"Unsupported data type for key {key}: {type(data)}"
                    )

    def _write_data(
        self,
        hdf: Union[h5py.File, h5py.Group],
        dataset_name: str,
        data: np.ndarray,
        chunk_row_size: int = 512,
        compression: str = "lzf",
    ) -> int:
        """
        Write data to HDF5 dataset with automatic resizing.

        Args:
            hdf: HDF5 file or group
            dataset_name: Name of the dataset
            data: Data to write
            chunk_row_size: Number of rows per chunk
            compression: Compression algorithm to use

        Returns:
            Total number of samples in the dataset after writing
        """
        n_additions = data.shape[0]

        if dataset_name not in hdf:
            # Create new dataset with unlimited first dimension
            hdf.create_dataset(
                dataset_name,
                data=data,
                compression=compression,
                chunks=(chunk_row_size, *data.shape[1:]),
                maxshape=(None, *data.shape[1:]),
            )
            n_samples = n_additions
        else:
            # Append to existing dataset
            ds = hdf[dataset_name]
            n_existing = ds.shape[0]
            ds.resize((n_existing + n_additions), axis=0)
            ds[-n_additions:] = data
            n_samples = n_existing + n_additions

        return n_samples

    def write_tl_data(self, collected_data: Dict[int, Dict[str, np.ndarray]]):
        """Save collected training data to HDF5 file"""

        with h5py.File(self.hdf_file_path, mode="a") as hf:

            if TL_DATA_GROUP in hf:
                del hf[TL_DATA_GROUP]

            tl_data_group = hf.create_group(TL_DATA_GROUP)

            for seq_len, data_dict in collected_data.items():
                if not data_dict:
                    continue

                group_key = str(seq_len)
                grp = tl_data_group.create_group(group_key)

                for array_name, array_data in data_dict.items():
                    grp.create_dataset(
                        array_name,
                        data=array_data,
                        chunks=(1, *array_data.shape[1:]),
                    )

    @staticmethod
    def compute_id_statistics(
        pmsm_df: pl.DataFrame, q_value_cutoff, global_fdr: bool = False
    ) -> pd.DataFrame:

        target_df = pmsm_df.filter(pl.col("is_decoy") == False)

        prefix = "global_" if global_fdr else ""
        col_map = {
            "precursors": (f"{prefix}precursor_q_value", "precursor_index"),
            "peptides": (f"{prefix}peptide_q_value", "peptidoform_index"),
            "protein_groups": (f"{prefix}precursor_q_value", "protein_group"),
        }

        counts = {
            key: target_df.filter(pl.col(fdr_col) <= q_value_cutoff)[idx_col].n_unique()
            for key, (fdr_col, idx_col) in col_map.items()
        }

        # precursor_fdr_col = "precursor_q_value"
        # peptide_fdr_col = "peptide_q_value"
        # protein_group_fdr_col = "precursor_q_value"
        # if global_fdr:
        #     precursor_fdr_col = "global" + precursor_fdr_col
        #     peptide_fdr_col = "global" + peptide_fdr_col
        #     protein_group_fdr_col = "global" + protein_group_fdr_col

        # counts = {
        #     "precursors": target_df.filter(pl.col(precursor_fdr_col) <= q_value_cutoff)[
        #         "precursor_index"
        #     ].n_unique(),
        #     "peptides": target_df.filter(pl.col(peptide_fdr_col) <= q_value_cutoff)[
        #         "peptidoform_index"
        #     ].n_unique(),
        #     "protein_groups": target_df.filter(
        #         pl.col(protein_group_fdr_col) <= q_value_cutoff
        #     )["protein_group"].n_unique(),
        # }

        return counts


def test():

    result_mgr = ResultManager("test_run", Path(r"D:\benchmark\DIA\2022-HGSOC\delpi"))

    result_mgr.write_df(df, key="meta_df")

    meta_df = result_mgr.read_df(key="meta_df")
