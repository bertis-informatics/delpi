"""
MS2 Spectrum Extractor for DelPi Search Results

Given a pre-filtered pmsm_results DataFrame (or path to pmsm_results.parquet)
for a single run and the corresponding raw LC-MS/MS file path, this module
extracts MS2 spectra for the identified peptides and writes them in the same
HDF5 format used by the Prospect-based pre-training pipeline.

The :class:`DelPiResultMS2Extractor` extends
:class:`~delpi.search.tl.data_prep.TransferLearningDataPreparator` and reuses
all of its sequence-length-batched intensity-extraction logic.  Only the
file-level orchestration (raw-file loading, HDF5 append) is new.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Union

import h5py
import numpy as np
import polars as pl

from pymsio import ReaderFactory

from delpi.search.tl.data_prep import (
    TransferLearningConfig,
    TransferLearningDataPreparator,
)

logger = logging.getLogger(__name__)


class DelPiResultMS2Extractor(TransferLearningDataPreparator):
    """Extract MS2 spectra from DelPi search results and save to HDF5.

    Extends :class:`TransferLearningDataPreparator` so that the entire
    intensity-extraction pipeline (fragmentation setup, prefix-mass arrays,
    numba kernels) is shared without duplication.

    The caller is responsible for pre-filtering *pmsm_df* (e.g. FDR cutoff,
    decoy removal) before passing it in.

    Example
    -------
    >>> config = TransferLearningConfig(tolerance_in_ppm=10.0)
    >>> extractor = DelPiResultMS2Extractor(config)
    >>> extractor.extract_and_save(
    ...     pmsm_df=filtered_df,
    ...     raw_file_path="/data/raw/run01.mzml",
    ...     h5_file_path="/data/tl_ms2.h5",
    ... )
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract_and_save(
        self,
        pmsm_df: Union[pl.DataFrame, Path, str],
        raw_file_path: Union[Path, str],
        h5_file_path: Union[Path, str],
    ) -> None:
        """Extract MS2 spectra for a single run and append to an HDF5 file.

        Parameters
        ----------
        pmsm_df:
            A pre-filtered Polars DataFrame **or** a path to a parquet file
            containing PSMs for the given run.
            Required columns: ``precursor_index``, ``frame_num``, ``peptide``,
            ``mod_ids``, ``mod_sites``, ``precursor_charge``,
            ``sequence_length``.
        raw_file_path:
            Path to the raw LC-MS/MS file (``.raw``, ``.mzml``, or
            ``.mzml.gz``) for this run.
        h5_file_path:
            Destination HDF5 file.  Created if it does not exist; existing
            datasets are extended (append mode).
        """
        raw_file_path = Path(raw_file_path)
        h5_file_path = Path(h5_file_path)
        h5_file_path.parent.mkdir(parents=True, exist_ok=True)

        # ── load DataFrame ────────────────────────────────────────────
        if isinstance(pmsm_df, (str, Path)):
            logger.info(f"Loading PMSM results from {pmsm_df}")
            pmsm_df = pl.read_parquet(pmsm_df)

        if len(pmsm_df) == 0:
            logger.warning("pmsm_df is empty — nothing to extract.")
            return

        logger.info(
            f"Extracting MS2 spectra for {len(pmsm_df)} PSMs from {raw_file_path}"
        )

        reader = ReaderFactory.get_reader(raw_file_path)
        lcms_data = reader.load()

        collected_data = self._collect_ms2_intensity_data(pmsm_df, lcms_data)
        self.save_to_hdf(collected_data, h5_file_path)
        logger.info(f"Appended MS2 data → {h5_file_path}")

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_to_hdf(
        collected_data: Dict[int, Dict[str, np.ndarray]],
        h5_file_path: Path,
    ) -> None:
        """Append *collected_data* to an HDF5 file.

        The file layout mirrors the Prospect pre-training format:
        ``/<seq_len>/<array_name>`` with LZF compression and unlimited
        first-axis for incremental appending.

        Parameters
        ----------
        collected_data:
            Mapping ``{sequence_length: {array_name: np.ndarray}}``, as
            returned by :meth:`_collect_ms2_intensity_data`.
        h5_file_path:
            Destination HDF5 file (opened in append mode).
        """
        with h5py.File(h5_file_path, "a") as hf:
            for seq_len, data_dict in collected_data.items():
                if not data_dict:
                    continue
                grp = hf.require_group(str(seq_len))

                for array_name, array_data in data_dict.items():
                    if array_data.ndim == 1:
                        chunk_shape = (1,)
                        max_shape = (None,)
                    else:
                        chunk_shape = (1, *array_data.shape[1:])
                        max_shape = (None, *array_data.shape[1:])

                    if array_name not in grp:
                        grp.create_dataset(
                            array_name,
                            data=array_data,
                            compression="lzf",
                            chunks=chunk_shape,
                            maxshape=max_shape,
                        )
                    else:
                        ds = grp[array_name]
                        n_existing = ds.shape[0]
                        n_new = array_data.shape[0]
                        ds.resize(n_existing + n_new, axis=0)
                        ds[-n_new:] = array_data
