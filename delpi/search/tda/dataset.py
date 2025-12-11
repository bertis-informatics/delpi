from typing import List, Tuple, Dict, Callable, Optional
from pathlib import Path
import h5py

import polars as pl
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """Handles splitting of datasets for training and testing."""

    def __init__(self, test_size: float = 0.5, random_state: int = 928):
        self.test_size = test_size
        self.random_state = random_state

    def split_by_peptide(
        self, pmsm_df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split dataset by peptide to avoid data leakage.

        Args:
            pmsm_df: DataFrame with peptide-spectrum matches

        Returns:
            Tuple of (training dataframe, testing dataframe)
        """
        target_id_df = (
            pmsm_df.filter(pl.col("is_decoy") == False)
            .select(pl.col("peptide_index"))
            .unique(maintain_order=True)
        )
        decoy_id_df = (
            pmsm_df.filter(pl.col("is_decoy") == True)
            .select(pl.col("peptide_index"))
            .unique(maintain_order=True)
        )

        train_decoy_id_df, test_decoy_id_df = train_test_split(
            decoy_id_df,
            test_size=self.test_size,
            shuffle=True,
            random_state=self.random_state,
        )

        train_id_df = pl.concat((target_id_df, train_decoy_id_df))
        test_id_df = pl.concat((target_id_df, test_decoy_id_df))

        train_df = pmsm_df.join(train_id_df, on=["peptide_index"], how="inner")
        test_df = pmsm_df.join(test_id_df, on=["peptide_index"], how="inner")

        return train_df, test_df


class PMSMDataset(Dataset):
    """
    각 bag = (run_index, precursor_index)
    grouped_pmsm_df:
        columns = ["run_index", "precursor_index", "pmsm_indices"]
        pmsm_indices : List[int] (polars list column)
    feature_fetcher:
        Callable(run_index:int, pmsm_indices:List[int]) -> torch.Tensor[n_i, d]
    """

    PMSM_EMBEDDING_DIM = 193

    def __init__(
        self,
        pmsm_df: pl.DataFrame,
        hdf_files: List[Path],
        hdf_group_key: str,
        rt_scale: float = 1000.0,
        data_dict: Optional[Dict] = None,
    ):
        self.pmsm_df = pmsm_df
        self.hdf_files = hdf_files
        self._hfs = [None] * len(hdf_files)  # For file-based access
        self.group_key = hdf_group_key
        self.data_dict = data_dict
        self.use_memory = data_dict is not None
        self.rt_scale = rt_scale

    def __len__(self):
        return self.pmsm_df.shape[0]

    def __del__(self):
        if hasattr(self, "_hfs"):
            for i, hf in enumerate(self._hfs):
                if hf is not None and isinstance(hf, h5py.File):
                    hf.close()
                    self._hfs[i] = None

    def _get_hf(self, hdf_index):
        """Get HDF5 file handle for file-based access"""
        if self._hfs[hdf_index] is None:
            self._hfs[hdf_index] = h5py.File(self.hdf_files[hdf_index], "r")
        return self._hfs[hdf_index]

    def get_pmsm_embedding(
        self, run_index: int, pmsm_index: int, rt_diff: float
    ) -> torch.Tensor:
        """Fetch features for given run and PMSM indices"""

        pmsm_embedding = np.empty(self.PMSM_EMBEDDING_DIM, dtype=np.float32)
        pmsm_embedding[-1] = rt_diff / self.rt_scale

        if self.use_memory:
            pmsm_embedding[:-1] = self.data_dict[run_index][pmsm_index]
        else:
            hf = self._get_hf(run_index)
            features_group = hf[self.group_key]["features"]
            pmsm_embedding[:-1] = features_group[pmsm_index, ...]

        return torch.from_numpy(pmsm_embedding).float()

    def __getitem__(self, index: int):

        pmsm_df = self.pmsm_df
        run_index = pmsm_df.item(index, "run_index")
        pmsm_index = pmsm_df.item(index, "pmsm_index")
        rt_diff = pmsm_df.item(index, "observed_rt") - pmsm_df.item(
            index, "predicted_rt"
        )
        x_feature = self.get_pmsm_embedding(run_index, pmsm_index, rt_diff)
        y_label = torch.FloatTensor([pmsm_df.item(index, "is_decoy") == False])

        return x_feature, y_label

    def to_tensor_dataset(self) -> TensorDataset:
        """Convert to TensorDataset for DataLoader usage."""

        return self.create_tensor_dataset(self.pmsm_df, self.data_dict, self.rt_scale)

    @classmethod
    def create_tensor_dataset(
        cls,
        pmsm_df: pl.DataFrame,
        data_dict: Dict[int, np.ndarray],
        rt_scale: float = 1000.0,
    ) -> TensorDataset:

        x = np.empty((pmsm_df.shape[0], cls.PMSM_EMBEDDING_DIM), dtype=np.float32)
        temp_df = pmsm_df.select(
            pl.col("run_index", "pmsm_index", "observed_rt", "predicted_rt")
        ).with_row_index("_index")

        for run_idx, sub_df in temp_df.group_by("run_index"):
            indices = sub_df["_index"]
            embedding_arrs = data_dict[run_idx[0]]
            x[indices, :-1] = embedding_arrs[sub_df["pmsm_index"], :]
            x[indices, -1] = (sub_df["observed_rt"] - sub_df["predicted_rt"]) / rt_scale

        x = torch.from_numpy(x)
        y = (~pmsm_df["is_decoy"].to_torch()).float()
        return TensorDataset(x, y.reshape(-1, 1))
