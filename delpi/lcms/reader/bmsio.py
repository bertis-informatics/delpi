import os
import sys
import stat
import platform
import tqdm
import platform
from typing import Union, Sequence, List
from pathlib import Path

import polars as pl
import numpy as np

from delpi.lcms.reader.base import MassSpecFileReader, MassSpecData
from delpi import PROJECT_DIR


def get_server_executable():

    os_type = platform.system()
    bmsio_dir = PROJECT_DIR / "bmsio/bin"

    if os_type == "Windows":
        return bmsio_dir / "win-x64" / "BertisMsioServer.exe"
    elif os_type == "Linux":
        return bmsio_dir / "linux-x64" / "BertisMsioServer"
    elif os_type == "Darwin":
        # return bmsio_dir / 'osx-m1' / 'BertisMsioServer.exe'
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def ensure_executable_permissions(executable_path):
    """
    Ensure the executable has proper permissions

    This is the safest method because:
    1. Package managers don't preserve file permissions
    2. Wheel format doesn't store permission bits
    3. Cross-platform compatibility issues
    4. Runtime verification is most reliable
    """
    if not executable_path.exists():
        return False

    if platform.system() == "Windows":
        # Windows doesn't use Unix-style permissions
        return True

    try:
        current_permissions = executable_path.stat().st_mode
        # Add execute permissions: owner, group, others
        new_permissions = (
            current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )
        executable_path.chmod(new_permissions)
        return True
    except (OSError, PermissionError) as e:
        print(f"Warning: Could not set executable permissions: {e}", file=sys.stderr)
        return False


def start_bmsio_server():
    from bmsio.grpc_service.manager import BmsioGrpcService

    executable = get_server_executable()

    if not executable.exists():
        print(f"Error: Server executable not found at {executable}", file=sys.stderr)
        print(f"Expected platform: {platform.system()}", file=sys.stderr)
        sys.exit(1)

    # 🔒 Critical: Ensure executable permissions at runtime
    if not ensure_executable_permissions(executable):
        print(
            f"Error: Could not set executable permissions for {executable}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify permissions worked
    if not os.access(executable, os.X_OK):
        print(
            f"Error: Executable {executable} is not executable after permission setting",
            file=sys.stderr,
        )
        sys.exit(1)

    bmsio_server = BmsioGrpcService(executable)
    bmsio_server.start()
    if bmsio_server.server_address is None:
        raise RuntimeError("Failed to start BMSIO GRPC service")

    return bmsio_server


class BMSIOReader(MassSpecFileReader):

    def __init__(
        self,
        file_path: Union[str, Path],
        num_workers: int = 0,
        server_addr: str = "localhost:5001",
    ):
        super().__init__(file_path, num_workers)
        self._reader = None
        self._meta_df = None
        self._server_addr = server_addr

    @property
    def reader(self):
        from bmsio.mass_spec import MsFileReader

        if self._reader is None:
            if not self.file_path.exists():
                raise FileNotFoundError(f"Cannot find `{self.file_path}`")
            self._reader = MsFileReader(server_address=self._server_addr)
            _ = self.reader.open(self.file_path)

        return self._reader

    def get_meta_df(self) -> pl.DataFrame:
        if self._meta_df is None:
            meta_df, iso_win_df = self.reader.get_meta_info()
            meta_df = meta_df[["mz_lo", "mz_hi", "time_in_seconds", "ms_level"]].join(
                iso_win_df[["isolation_min_mz", "isolation_max_mz"]]
            )

            meta_df = pl.from_pandas(
                meta_df, include_index=True, schema_overrides=self.meta_schema
            )
            self._meta_df = meta_df

        return self._meta_df

    def get_frame(self, frame_num: int) -> np.ndarray:

        frame = self.reader.read_frame(frame_num)
        peaks = np.column_stack(
            [frame.payload.as_arrays.mz, frame.payload.as_arrays.ab]
        ).astype(np.float32)

        return peaks

    def get_frames(self, frame_nums: Sequence[int]) -> List[np.ndarray]:
        spectra = list()
        for frame in self.reader.iterate_frames(frame_nums=frame_nums):
            spectra.append(
                np.column_stack(
                    [frame.payload.as_arrays.mz, frame.payload.as_arrays.ab]
                ).astype(np.float32)
            )
        return spectra

    def load(self) -> MassSpecData:

        meta_df = self.get_meta_df()
        min_frame_num = meta_df.item(0, "frame_num")
        max_frame_num = meta_df.item(-1, "frame_num")

        def read_spectra(min_frame_num, max_frame_num):
            spectra = list()
            frame_itr = self._reader.iterate_frames(min_frame_num, max_frame_num)
            for frame in frame_itr:
                spectra.append(
                    np.column_stack(
                        [frame.payload.as_arrays.mz, frame.payload.as_arrays.ab]
                    ).astype(np.float32)
                )
            return spectra

        # load whole spectra into memory
        batch_size = 1024 * 10
        batch_ranges = [
            (start, min(start + batch_size - 1, max_frame_num))
            for start in range(min_frame_num, max_frame_num + 1, batch_size)
        ]

        all_spectra = list()
        for min_fr, max_fr in tqdm.tqdm(batch_ranges, desc="load spectra"):
            spectra = read_spectra(min_fr, max_fr)
            all_spectra.extend(spectra)

        return MassSpecData.create(self.run_name, meta_df, all_spectra)
