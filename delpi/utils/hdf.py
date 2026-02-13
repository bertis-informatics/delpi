import os
from pathlib import Path
from typing import List
import logging

import h5py
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HdfDataset(Dataset):
    """Base Dataset with lazy HDF5 file handle caching, safe for multiprocessing.

    Handles:
    - Lazy opening of HDF5 files on first access
    - Process-aware cache reset (fork/spawn safe via os.getpid check)
    - Pickle support for 'spawn' multiprocessing (__getstate__/__setstate__)
    - Automatic cleanup on deletion
    """

    def __init__(self, hdf_files: List[Path]):
        if not isinstance(hdf_files, list):
            hdf_files = [hdf_files]
        self.hdf_files = hdf_files
        self._hfs: List = [None] * len(hdf_files)
        self._pid: int = os.getpid()

    def get_hf(self, hdf_index: int) -> h5py.File:
        """Get cached HDF5 file handle, resetting if in a new process."""
        pid = os.getpid()
        if self._pid != pid:
            self._hfs = [None] * len(self.hdf_files)
            self._pid = pid
        if self._hfs[hdf_index] is None:
            self._hfs[hdf_index] = h5py.File(self.hdf_files[hdf_index], "r")
        return self._hfs[hdf_index]

    def __del__(self):
        if hasattr(self, "_hfs"):
            for hf in self._hfs:
                if hf is not None and isinstance(hf, h5py.File):
                    hf.close()

    def __getstate__(self):
        """Pickle support for 'spawn' multiprocessing (h5py handles can't be pickled)."""
        state = self.__dict__.copy()
        state["_hfs"] = [None] * len(self.hdf_files)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
