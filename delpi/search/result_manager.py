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

    # def write_df(
    #     self,
    #     df: pl.DataFrame,
    #     key: str,
    #     complib: str = "blosc:zstd",
    #     complevel: int = 4,
    # ) -> None:
    #     df.to_pandas().to_hdf(
    #         self.hdf_file_path,
    #         key=key,
    #         mode="a",
    #         format="fixed",
    #         complib=complib,
    #         complevel=complevel,
    #     )

    # def read_df(self, key: str) -> pl.DataFrame:
    #     return pl.from_pandas(pd.read_hdf(self.hdf_file_path, key=key))

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

    def load_features(
        self,
        group_key: str,
        feature_dim: int,
        out: np.ndarray = None,
        row_indices: np.ndarray = None,
    ) -> np.ndarray:
        """Read embedding features from HDF into a (N, feature_dim) array.

        Parameters
        ----------
        group_key : str
            HDF group name (e.g. "second_results").
        feature_dim : int
            Width of the output array (embedding + extra columns).
        out : np.ndarray, optional
            Pre-allocated destination array.  When provided, embeddings are
            written into the rows given by *row_indices* (or all rows if
            *row_indices* is ``None``).  Columns beyond the embedding dim
            are left untouched.
        row_indices : np.ndarray, optional
            HDF row indices to read (fancy-indexing into the dataset).
            Only used when *out* is provided.

        Returns
        -------
        np.ndarray
            The filled array — either *out* or a freshly allocated one.
        """
        with h5py.File(self.hdf_file_path, mode="r") as hdf_file:
            ds = hdf_file[group_key]["features"]
            embed_dim = ds.shape[1]

            if out is None:
                n = ds.shape[0]
                out = np.empty((n, feature_dim), dtype=np.float32)
                ds.read_direct(out, dest_sel=np.s_[:, :embed_dim])
            elif row_indices is not None:
                all_embeddings = ds[:]
                out[:, :embed_dim] = all_embeddings[row_indices]
            else:
                ds.read_direct(out, dest_sel=np.s_[:, :embed_dim])

        return out

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
            chunk_shape = (chunk_row_size, *data.shape[1:])
            # Disable compression for pathological tiny-chunk 1D case
            use_compression = compression
            if data.ndim == 1 and chunk_row_size == 1:
                use_compression = None  # disables compression in h5py

            # Create new dataset with unlimited first dimension
            hdf.create_dataset(
                dataset_name,
                data=data,
                compression=use_compression,
                chunks=chunk_shape,
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

    @staticmethod
    def compute_id_statistics(
        pmsm_df: pl.DataFrame,
        q_value_cutoff,
        global_fdr: bool = False,
        use_library_q_value: bool = False,
    ) -> pd.DataFrame:
        """Count identified precursors/peptides/protein-groups at *q_value_cutoff*.

        ``use_library_q_value`` is set for the second pass of the two-pass
        MBR search: the first-pass-derived ``library_*_q_value`` columns
        (fixed per precursor, not run- or global-scope specific) are used
        instead of ``global_fdr``'s ``global_``/run-specific column prefix.
        """

        target_df = pmsm_df.filter(pl.col("is_decoy") == False)

        if use_library_q_value:
            col_map = {
                "precursors": ("library_precursor_q_value", "precursor_index"),
                "peptides": ("library_peptide_q_value", "peptidoform_index"),
                "protein_groups": (
                    "library_protein_group_q_value",
                    "protein_group",
                ),
            }
        else:
            prefix = "global_" if global_fdr else ""
            col_map = {
                "precursors": (f"{prefix}precursor_q_value", "precursor_index"),
                "peptides": (f"{prefix}peptide_q_value", "peptidoform_index"),
                "protein_groups": (
                    f"{prefix}protein_group_q_value",
                    "protein_group",
                ),
            }

        # Columns may be absent (e.g. the second pass of the two-pass MBR
        # search skips protein grouping/its q-value); report `None` for
        # those metrics instead of raising.
        counts = {
            key: (
                target_df.filter(pl.col(fdr_col) <= q_value_cutoff)[idx_col].n_unique()
                if fdr_col in target_df.columns and idx_col in target_df.columns
                else None
            )
            for key, (fdr_col, idx_col) in col_map.items()
        }

        return counts
