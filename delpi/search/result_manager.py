"""Per-run HDF5 storage of search results."""

from pathlib import Path
from typing import Union, List, Dict
import logging

import h5py
import numpy as np
import polars as pl
import pandas as pd

logger = logging.getLogger(__name__)


class ResultManager:
    """Reads and writes one run's results in ``<output_dir>/<run_name>.delpi.h5``."""

    def __init__(self, run_name: str, output_dir: Path):
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
        """Read `data_keys` from `group_key`; keys absent from the group are skipped.

        Raises:
            KeyError: the group does not exist.
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
        """Read the ``features`` embeddings of a group into a (N, feature_dim) array.

        Args:
            feature_dim: output width (embedding + extra columns). Columns past
                the embedding are left untouched.
            out: preallocated destination. With `row_indices`, row k of `out`
                gets HDF row ``row_indices[k]``; otherwise all rows are read.
            row_indices: HDF rows to read; used only with `out`.

        Returns:
            `out`, or a new float32 array.
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
        """Append each array to its dataset in `group_key`, creating it if needed."""

        with h5py.File(self.hdf_file_path, mode="a") as hdf_file:

            if group_key in hdf_file:
                result_group = hdf_file[group_key]
            else:
                result_group = hdf_file.create_group(group_key)

            for key, data in data_dict.items():
                # one-row chunks give fast random access to features in training
                chunk_row_size = 1 if key == "features" else chunk_size
                if isinstance(data, np.ndarray):
                    self._write_data(result_group, key, data, chunk_row_size)
                elif hasattr(data, "__array__"):
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
        """Append `data` to a dataset, creating a resizable one if missing.

        Returns:
            Number of rows in the dataset after writing.
        """
        n_additions = data.shape[0]

        if dataset_name not in hdf:
            chunk_shape = (chunk_row_size, *data.shape[1:])
            # no compression for 1-D data in one-row chunks
            use_compression = compression
            if data.ndim == 1 and chunk_row_size == 1:
                use_compression = None

            hdf.create_dataset(
                dataset_name,
                data=data,
                compression=use_compression,
                chunks=chunk_shape,
                maxshape=(None, *data.shape[1:]),
            )
            n_samples = n_additions
        else:
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
        """Count target precursors, peptides and protein groups at `q_value_cutoff`.

        `use_library_q_value` (second pass of the two-pass MBR search) uses the
        first-pass ``library_*_q_value`` columns, fixed per precursor, instead
        of the run-level (or, with `global_fdr`, ``global_``) columns. A count
        is None when its columns are absent, e.g. the second MBR pass skips
        protein grouping.
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

        counts = {
            key: (
                target_df.filter(pl.col(fdr_col) <= q_value_cutoff)[idx_col].n_unique()
                if fdr_col in target_df.columns and idx_col in target_df.columns
                else None
            )
            for key, (fdr_col, idx_col) in col_map.items()
        }

        return counts
