"""
HDF5 Utilities for DelPi

This module provides basic utilities for working with HDF5 files, including:
- Sparse array serialization/deserialization
- Data storage with efficient chunking and compression
"""

from typing import Union
import logging

import h5py
import numpy as np
from scipy.sparse import csr_array, csc_array

logger = logging.getLogger(__name__)


# Constants for supported array types
_SUPPORTED_ARRAYS = (np.ndarray, csr_array, csc_array)
_NAME_TO_ARRAY_CLASS = {arr_cls.__name__: arr_cls for arr_cls in _SUPPORTED_ARRAYS}
_NAME_TO_ARRAY_CLASS[np.ndarray.__name__] = csr_array


class SparseArrayHandler:
    """Handles serialization and deserialization of sparse arrays in HDF5."""

    @staticmethod
    def save(
        hdf_group: h5py.Group, array: Union[np.ndarray, csr_array, csc_array]
    ) -> h5py.Group:
        """
        Save a sparse array to HDF5 group.

        Args:
            hdf_group: HDF5 group to save the array to
            array: Array to save (will be converted to CSR if needed)

        Returns:
            The HDF5 group with saved data
        """
        if isinstance(array, np.ndarray):
            csr = (
                csr_array(array)
                if array.ndim == 2
                else csr_array(array.reshape(array.shape[0], -1))
            )
        elif isinstance(array, _SUPPORTED_ARRAYS):
            csr = array
        else:
            raise ValueError(f"Unsupported array type: {type(array)}")

        hdf_group.create_dataset("indices", data=csr.indices, dtype=np.int32)
        hdf_group.create_dataset("data", data=csr.data, dtype=array.dtype)
        hdf_group.create_dataset("indptr", data=csr.indptr, dtype=np.int32)
        hdf_group.attrs["shape"] = array.shape
        hdf_group.attrs["class"] = array.__class__.__name__

        return hdf_group

    @staticmethod
    def load(hdf_group: h5py.Group) -> Union[np.ndarray, csr_array, csc_array]:
        """
        Load a sparse array from HDF5 group.

        Args:
            hdf_group: HDF5 group containing the sparse array data

        Returns:
            Reconstructed array in original format
        """
        shape = hdf_group.attrs["shape"].tolist()
        arr_cls_name = hdf_group.attrs["class"]
        arr_cls = _NAME_TO_ARRAY_CLASS[arr_cls_name]

        csr = arr_cls(
            (
                np.array(hdf_group["data"]),
                np.array(hdf_group["indices"], dtype=np.int32),
                np.array(hdf_group["indptr"], dtype=np.int32),
            ),
            shape=(shape[0], np.prod(shape[1:])),
        )

        if arr_cls_name == np.ndarray.__name__:
            return csr.toarray().reshape(shape)

        return csr


def write_data(
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
        # Create new dataset
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
        n_existing = hdf[dataset_name].shape[0]
        hdf[dataset_name].resize((n_existing + n_additions), axis=0)
        hdf[dataset_name][-n_additions:] = data
        n_samples = n_existing + n_additions

    return n_samples
