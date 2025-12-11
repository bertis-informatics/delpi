from typing import Union, Sequence, List
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
import h5py

from delpi.lcms.reader.base import MassSpecFileReader, MassSpecData


class HdfReader(MassSpecFileReader):

    def __init__(
        self, file_path: Union[str, Path], run_name: str = None, num_workers: int = 0
    ):
        super().__init__(file_path, num_workers=num_workers)

        self._hf = h5py.File(self.file_path, "r")

        if run_name is None:
            run_name = self.get_run_names()[0]
        else:
            assert run_name in self.get_run_names()
        self._cur_run_name = run_name
        self._load_meta_df()

    def _load_meta_df(self):

        group_key = self._cur_run_name
        meta_df = pl.from_pandas(
            pd.read_hdf(self.file_path, key=f"{group_key}/meta_df")
        )

        self.frame_num_to_index = np.empty(
            meta_df.item(-1, "frame_num") + 1, dtype=np.uint32
        )
        for idx, num in enumerate(meta_df["frame_num"]):
            self.frame_num_to_index[num] = idx

        self._meta_df = meta_df

    @property
    def run_name(self) -> str:
        return self._cur_run_name

    def get_run_names(self):
        try:
            run_names = list(self._hf)
        except:
            return list()
        return run_names

    def get_meta_df(self) -> pl.DataFrame:
        return self._meta_df

    def load(self) -> MassSpecData:
        peak_arr = self._read_peaks()
        return MassSpecData(self.run_name, self._meta_df, peak_arr)

    def _read_peaks(self, indices=None) -> np.ndarray:
        group_key = self.run_name
        hf_grp = self._hf[group_key]
        if indices is None:
            peak_arr = np.asarray(hf_grp["peak"])
        else:
            peak_arr = np.asarray(hf_grp["peak"][indices])

        return peak_arr

    def get_frame(self, frame_num: int) -> np.ndarray:
        meta_df = self.get_meta_df()
        idx = self.frame_num_to_index[frame_num]
        st, ed = meta_df.item(idx, "peak_start"), meta_df.item(idx, "peak_stop")

        if (st is None) or (ed is None) or (ed <= st):
            return np.empty((0, 2), dtype=np.float32)

        return self._read_peaks(indices=slice(st, ed))

    def get_frames(self, frame_nums: Sequence[int]) -> List[np.ndarray]:
        return [self.get_frame(fn) for fn in frame_nums]
