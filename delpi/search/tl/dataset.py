from pathlib import Path
from typing import List, Optional, Dict

import torch
import polars as pl
import numpy as np
from torch.utils.data import Dataset

from delpi.model.spec_lib.aa_encoder import encode_modification_feature
from delpi.utils.hdf import HdfDataset

LABEL_DTYPE = np.dtype(
    [
        ("seq_len", np.int16),
        ("index", np.uint32),
        ("precursor_index", np.uint32),
    ]
)


class TransferLearningDataset(HdfDataset):
    """
    Unified Transfer Learning Dataset that automatically decides between memory and file access
    based on dataset size. Uses in-memory storage for datasets with < 1M samples.
    """

    def __init__(
        self,
        hdf_files: List[Path],
        labels: np.ndarray,
        data_dict: Optional[Dict] = None,
    ):
        super().__init__(hdf_files)
        # self.label_df = label_df
        self.data_dict = data_dict
        self.use_memory = data_dict is not None
        self.labels = labels  # numpy structured array with LABEL_DTYPE
        # Pre-convert to numpy for worker-safe __getitem__ (no polars in workers)
        # self._seq_lens = label_df["seq_len"].to_numpy()
        # self._indices = label_df["index"].to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        n_tokens = self.labels["seq_len"][index]
        idx = self.labels["index"][index]

        if self.use_memory:
            # Get data from pre-loaded numpy arrays (same structure as HDF)
            x_aa = self.data_dict[n_tokens]["x_aa"][idx]
            x_mod = self.data_dict[n_tokens]["x_mod"][idx]
            x_meta = self.data_dict[n_tokens]["x_meta"][idx]
            y_intensity = self.data_dict[n_tokens]["x_intensity"][idx]
        else:
            # Get data from the shared HDF5 file (single file, no per-run
            # grouping).
            hf = self.get_hf(0)
            data_group = hf[str(n_tokens)]

            x_aa = data_group["x_aa"][idx][...]
            x_mod = data_group["x_mod"][idx][...]
            x_meta = data_group["x_meta"][idx][...]
            y_intensity = data_group["x_intensity"][idx][...]

        # Apply transformations
        x_mod = encode_modification_feature(x_mod, x_aa.shape[0])

        return {
            "x_aa": x_aa,
            "x_mod": x_mod,
            "x_meta": x_meta,
            "y_intensity": y_intensity,
        }

    def make_subset(self, fractions, seed):
        rng = np.random.default_rng(seed)
        n = len(self.labels)
        size = int(n * fractions)
        subset_idx = rng.choice(n, size=size, replace=False)

        return self.__class__(
            self.hdf_files,
            labels=self.labels[subset_idx],
            data_dict=self.data_dict,
        )

    def make_subset_from_indices(self, indices: np.ndarray):
        """Build a subset using explicit sample indices.

        This is used by deterministic global-permutation chunk sampling so that
        all distributed workers consume the same sample pool each epoch.
        """
        return self.__class__(
            self.hdf_files,
            labels=self.labels[indices],
            data_dict=self.data_dict,
        )


class TransferLearningDatasetForRT(Dataset):

    def __init__(
        self,
        labels: np.ndarray,
        data_dict: Dict,
    ):
        self.labels = labels
        self.data_dict = data_dict
        # Pre-convert to numpy for worker-safe __getitem__ (no polars in workers)
        # self._seq_lens = label_df["seq_len"].to_numpy()
        # self._indices = label_df["index"].to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        n_tokens = self.labels["seq_len"][index]
        idx = self.labels["index"][index]

        # Get data from pre-loaded numpy arrays (same structure as HDF)
        x_aa = self.data_dict[n_tokens]["x_aa"][idx]
        x_mod = self.data_dict[n_tokens]["x_mod"][idx]
        y_rt = self.data_dict[n_tokens]["x_rt"][idx]

        # Apply transformations
        x_mod = encode_modification_feature(x_mod, x_aa.shape[0])

        return {
            "x_aa": x_aa,
            "x_mod": x_mod,
            "rt": torch.FloatTensor([y_rt]),
        }
