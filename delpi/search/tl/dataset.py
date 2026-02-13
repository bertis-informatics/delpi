from pathlib import Path
from typing import List, Optional, Dict

import torch
import polars as pl
from torch.utils.data import Dataset

from delpi.model.spec_lib.aa_encoder import encode_modification_feature
from delpi.search.result_manager import TL_DATA_GROUP
from delpi.utils.hdf import HdfDataset


class TransferLearningDataset(HdfDataset):
    """
    Unified Transfer Learning Dataset that automatically decides between memory and file access
    based on dataset size. Uses in-memory storage for datasets with < 1M samples.
    """

    def __init__(
        self,
        hdf_files: List[Path],
        label_df: pl.DataFrame,
        data_dict: Optional[Dict] = None,
    ):
        super().__init__(hdf_files)
        self.label_df = label_df
        self.data_dict = data_dict
        self.use_memory = data_dict is not None

        # Pre-convert to numpy for worker-safe __getitem__ (no polars in workers)
        self._seq_lens = label_df["seq_len"].to_numpy()
        self._indices = label_df["index"].to_numpy()
        self._fids = label_df["hdf_index"].to_numpy()

    def __len__(self):
        return len(self._fids)

    def __getitem__(self, index):
        n_tokens = self._seq_lens[index]
        idx = self._indices[index]
        fid = self._fids[index]

        if self.use_memory:
            # Get data from pre-loaded numpy arrays (same structure as HDF)
            x_aa = self.data_dict[fid][n_tokens]["x_aa"][idx]
            x_mod = self.data_dict[fid][n_tokens]["x_mod"][idx]
            x_meta = self.data_dict[fid][n_tokens]["x_meta"][idx]
            y_intensity = self.data_dict[fid][n_tokens]["x_intensity"][idx]
        else:
            # Get data from HDF5 files
            hf = self.get_hf(fid)
            tl_data_group = hf[TL_DATA_GROUP]
            data_group = tl_data_group[str(n_tokens)]

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
        """Create a subset of the dataset with the given fraction of data"""
        assert fractions > 0 and fractions <= 1, "Fraction must be in (0, 1]"

        new_label_df = self.label_df.sample(
            fraction=fractions, seed=seed, with_replacement=False
        )

        return self.__class__(
            self.hdf_files,
            label_df=new_label_df,
            data_dict=self.data_dict,
        )


class TransferLearningDatasetForRT(Dataset):

    def __init__(
        self,
        label_df: pl.DataFrame,
        data_dict: Dict,
    ):
        self.label_df = label_df
        self.data_dict = data_dict

        # Pre-convert to numpy for worker-safe __getitem__ (no polars in workers)
        self._seq_lens = label_df["seq_len"].to_numpy()
        self._indices = label_df["index"].to_numpy()
        self._fids = label_df["hdf_index"].to_numpy()

    def __len__(self):
        return len(self._fids)

    def __getitem__(self, index):
        n_tokens = self._seq_lens[index]
        idx = self._indices[index]
        fid = self._fids[index]

        # Get data from pre-loaded numpy arrays (same structure as HDF)
        x_aa = self.data_dict[fid][n_tokens]["x_aa"][idx]
        x_mod = self.data_dict[fid][n_tokens]["x_mod"][idx]
        y_rt = self.data_dict[fid][n_tokens]["x_rt"][idx]

        # Apply transformations
        x_mod = encode_modification_feature(x_mod, x_aa.shape[0])

        return {
            "x_aa": x_aa,
            "x_mod": x_mod,
            "rt": torch.FloatTensor([y_rt]),
        }
