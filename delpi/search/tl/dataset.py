from pathlib import Path
from typing import List, Optional, Dict
import h5py

import torch
import polars as pl
from torch.utils.data import Dataset

from delpi.model.spec_lib.aa_encoder import encode_modification_feature
from delpi.search.result_manager import TL_DATA_GROUP


class TransferLearningDataset(Dataset):
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
        if not isinstance(hdf_files, List):
            hdf_files = [hdf_files]

        self.hdf_files = hdf_files
        self._hfs = [None] * len(hdf_files)  # For file-based access
        self.label_df = label_df
        self.data_dict = data_dict
        self.use_memory = data_dict is not None

    def __len__(self):
        return self.label_df.shape[0]

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

    def __getitem__(self, index):
        n_tokens = self.label_df.item(index, "seq_len")
        idx = self.label_df.item(index, "index")
        fid = self.label_df.item(index, "hdf_index")

        if self.use_memory:
            # Get data from pre-loaded numpy arrays (same structure as HDF)
            x_aa = self.data_dict[fid][n_tokens]["x_aa"][idx]
            x_mod = self.data_dict[fid][n_tokens]["x_mod"][idx]
            x_meta = self.data_dict[fid][n_tokens]["x_meta"][idx]
            y_intensity = self.data_dict[fid][n_tokens]["x_intensity"][idx]
        else:
            # Get data from HDF5 files
            hf = self._get_hf(fid)
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

    def __len__(self):
        return self.label_df.shape[0]

    def __getitem__(self, index):
        n_tokens = self.label_df.item(index, "seq_len")
        idx = self.label_df.item(index, "index")
        fid = self.label_df.item(index, "hdf_index")

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
