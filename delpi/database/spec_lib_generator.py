import os
import uuid
from pathlib import Path
from typing import Union

import polars as pl
import numpy as np
import pyarrow.parquet as pq

import torch
from torch.utils.data import DataLoader

from delpi.lcms.base_ion_type import BaseIonType
from delpi.lcms.fragmentation import Fragmentation
from delpi.lcms.neutral_loss import NeutralLoss
from delpi.database.spec_lib_schema import (
    SPECLIB_SCHEMA,
    fragment_offsets,
    with_speclib_index,
)
from delpi.model.spec_lib import Ms2SpectrumPredictor, RetentionTimePredictor
from delpi.model.spec_lib.dataset import PeptideDataset
from delpi.utils.batch_sampler import SeqDataBatchSampler, ChunkedSeqDataBatchSampler
from delpi.database.numba.prefix_mass_array import PrefixMassArrayContainer
from delpi.database.numba.spec_lib_utils import update_speclib_chunk_arr
from delpi.utils.prefetch import Prefetcher, pin_tensor_dict
from delpi.search.progress import ProgressTracker, TqdmProgressTracker
from delpi import MODEL_DIR


class SpectralLibGenerator:
    """Generate libraries with the top N fragments by predicted intensity.

    min_intensity is the inclusive normalized-intensity threshold and
    max_fragments is an upper bound. Rows are stored in increasing intensity
    order, with rank 1 assigned to the strongest fragment.
    """

    speclib_df_schema = SPECLIB_SCHEMA

    def __init__(
        self,
        min_charge: int = 1,
        max_charge: int = 2,
        prefix_ion_type=BaseIonType.B,
        suffix_ion_type=BaseIonType.Y,
        max_fragments: int = 16,
        min_fragment_mz: float = 200,
        max_fragment_mz: float = 1800,
        apply_phospho: bool = False,
        min_intensity: float = 0.01,
        device: Union[str, torch.device] = "cuda:0",
        ms2_predictor: Ms2SpectrumPredictor = None,
        rt_predictor: RetentionTimePredictor = None,
    ):
        if not isinstance(max_fragments, int) or not 1 <= max_fragments <= 255:
            raise ValueError("max_fragments must be an integer in [1, 255].")
        if not 0 <= min_intensity <= 1:
            raise ValueError("min_intensity must be in [0, 1].")
        self.max_fragments = max_fragments
        self.min_intensity = min_intensity
        self.min_fragment_mz = min_fragment_mz
        self.max_fragment_mz = max_fragment_mz
        neutral_losses = [NeutralLoss.NO_LOSS]
        if apply_phospho:
            neutral_losses.append(NeutralLoss.H3O4P)

        self.fragmentation = Fragmentation(
            min_charge,
            max_charge,
            prefix_ion_type=prefix_ion_type,
            suffix_ion_type=suffix_ion_type,
            max_fragment_isotopes=1,
            neutral_losses=neutral_losses,
        )

        self.device = device
        self._ms2_predictor = ms2_predictor
        self._rt_predictor = rt_predictor

    @property
    def ms2_predictor(self):
        if self._ms2_predictor is None:
            self._ms2_predictor = Ms2SpectrumPredictor.load(
                MODEL_DIR / "delpi.ms2_predictor.pth"
            )
        return self._ms2_predictor.to(self.device).eval()

    @property
    def rt_predictor(self):
        if self._rt_predictor is None:
            self._rt_predictor = RetentionTimePredictor.load(
                MODEL_DIR / "delpi.rt_predictor.pth"
            )
        return self._rt_predictor.to(self.device).eval()

    @property
    def param_dict(self):
        return {
            "min_charge": self.fragmentation.min_charge,
            "max_charge": self.fragmentation.max_charge,
            "prefix_ion_type": self.fragmentation.base_ion_types[0].symbol,
            "suffix_ion_type": self.fragmentation.base_ion_types[1].symbol,
            "max_fragments": self.max_fragments,
            "min_intensity": self.min_intensity,
            # 2: variable fragment rows per precursor (speclib_offset/rows).
            "format_version": 2,
            "min_mz": self.min_fragment_mz,
            "max_mz": self.max_fragment_mz,
        }

    @staticmethod
    def _compact_speclib(arrays, counts, max_fragments):
        capacity = len(counts) * max_fragments
        keep = (np.arange(max_fragments)[None, :] < counts[:, None]).ravel()
        data = {name: values[:capacity][keep] for name, values in arrays.items()}
        return pl.DataFrame(data, schema=SPECLIB_SCHEMA)

    def predict_ms2_spectra(
        self,
        peptide_df: pl.DataFrame,
        modification_df: pl.DataFrame,
        precursor_df: pl.DataFrame,
        prefix_mass_container: PrefixMassArrayContainer,
        batch_size: int = 512,
        detectable_min_mz: float = None,
        detectable_max_mz: float = None,
        progress: ProgressTracker = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return (speclib_df, precursor_df with offsets and row counts)."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        n = precursor_df.height
        if n == 0:
            return pl.DataFrame(schema=SPECLIB_SCHEMA), with_speclib_index(
                precursor_df, np.zeros(0, dtype=np.uint8)
            )
        own_progress = progress is None
        if own_progress:
            progress = TqdmProgressTracker(
                total=n, description="Predicting MS2 spectra"
            )
            batch_progress = progress
        else:
            batch_progress = progress.create_child(
                "Predicting MS2 spectra", total=n, portion=100
            )
        frames = []

        class MemoryWriter:
            def write_table(self, table, **kwargs):
                frames.append(pl.from_arrow(table))

        try:
            counts = self._write_ms2_spectra_chunks(
                MemoryWriter(),
                peptide_df,
                modification_df,
                precursor_df,
                prefix_mass_container,
                batch_size,
                n,
                (
                    self.min_fragment_mz
                    if detectable_min_mz is None
                    else detectable_min_mz
                ),
                (
                    self.max_fragment_mz
                    if detectable_max_mz is None
                    else detectable_max_mz
                ),
                batch_progress,
            )
            batch_progress.complete()
        finally:
            if own_progress:
                progress.close()
        speclib_df = (
            pl.concat(frames) if frames else pl.DataFrame(schema=SPECLIB_SCHEMA)
        )
        return speclib_df, with_speclib_index(precursor_df, counts)

    def predict_ms2_spectra_to_parquet(
        self,
        peptide_df: pl.DataFrame,
        modification_df: pl.DataFrame,
        precursor_df: pl.DataFrame,
        prefix_mass_container: PrefixMassArrayContainer,
        save_dir: Union[str, Path],
        batch_size: int = 1024,
        chunk_size: int = 65_536,
        detectable_min_mz: float = None,
        detectable_max_mz: float = None,
        progress: ProgressTracker = None,
    ) -> tuple[Path, pl.DataFrame]:
        """Write save_dir/speclib_df.parquet; return (path, indexed precursor_df).

        The caller saves precursor_df with the other tables (PeptideDatabase.save).
        Only one chunk of output rows (chunk_size * max_fragments) is held in
        memory; the input pipeline is still built once for the whole run.
        """

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(save_dir) / "speclib_df.parquet"
        n_precursors = precursor_df.shape[0]

        arrow_schema = pl.DataFrame(schema=self.speclib_df_schema).to_arrow().schema

        # One bar for the whole task. A caller's tracker has an arbitrary total,
        # so a child sized to the precursor count is driven instead. Progress
        # counts precursors because a batch count would need a sampler pre-pass.
        own_progress = progress is None
        if own_progress:
            progress = TqdmProgressTracker(
                total=max(n_precursors, 1), description="Predicting MS2 spectra"
            )
            batch_progress = progress
        else:
            batch_progress = progress.create_child(
                "Predicting MS2 spectra", total=max(n_precursors, 1), portion=100
            )

        tmp_path = output_path.with_name(f"{output_path.name}.tmp-{uuid.uuid4().hex}")
        writer = pq.ParquetWriter(tmp_path, schema=arrow_schema, compression="zstd")
        counts = np.zeros(n_precursors, dtype=np.uint8)
        try:
            if n_precursors > 0:
                counts = self._write_ms2_spectra_chunks(
                    writer,
                    peptide_df=peptide_df,
                    modification_df=modification_df,
                    precursor_df=precursor_df,
                    prefix_mass_container=prefix_mass_container,
                    batch_size=batch_size,
                    chunk_size=chunk_size,
                    detectable_min_mz=(
                        self.min_fragment_mz
                        if detectable_min_mz is None
                        else detectable_min_mz
                    ),
                    detectable_max_mz=(
                        self.max_fragment_mz
                        if detectable_max_mz is None
                        else detectable_max_mz
                    ),
                    progress=batch_progress,
                )
            writer.close()
            writer = None
            indexed_precursor_df = with_speclib_index(precursor_df, counts)
            os.replace(tmp_path, output_path)
            if own_progress:
                progress.complete()
            else:
                batch_progress.complete()
        finally:
            if writer is not None:
                writer.close()
            if tmp_path.exists():
                tmp_path.unlink()
            if own_progress:
                progress.close()

        return output_path, indexed_precursor_df

    def _write_ms2_spectra_chunks(
        self,
        writer: pq.ParquetWriter,
        peptide_df: pl.DataFrame,
        modification_df: pl.DataFrame,
        precursor_df: pl.DataFrame,
        prefix_mass_container: PrefixMassArrayContainer,
        batch_size: int,
        chunk_size: int,
        detectable_min_mz: float,
        detectable_max_mz: float,
        progress: ProgressTracker,
    ):
        max_fragments = self.max_fragments
        n_precursors = precursor_df.shape[0]

        # chunk boundaries and update_speclib_chunk_arr's local indices assume
        # precursor_index == row position
        self._assert_contiguous_precursor_index(precursor_df, 0, n_precursors)

        # indexed by global precursor_index
        peptidoform_index_arr = precursor_df["peptidoform_index"].to_numpy()
        ion_type_container = self.fragmentation.get_ion_types()

        # batches are grouped by sequence_length and never span two chunks, so
        # chunk transitions can be detected from the batch stream
        precursor_ds = PeptideDataset(
            precursor_df, modification_df, peptide_df, level="precursor"
        )
        batch_sampler = ChunkedSeqDataBatchSampler(
            precursor_ds,
            batch_size=batch_size,
            chunk_size=chunk_size,
            batch_grouping_column="sequence_length",
        )
        dl = DataLoader(
            dataset=precursor_ds, batch_sampler=batch_sampler, num_workers=0
        )

        max_token_len = peptide_df["sequence_length"].max() + 3
        mod_feat_dim = self.ms2_predictor.mod_embedding.in_features
        X_aa_buf = torch.empty(
            (batch_size, max_token_len), dtype=torch.long, device=self.device
        )
        X_mod_buf = torch.empty(
            (batch_size, max_token_len, mod_feat_dim),
            dtype=torch.float32,
            device=self.device,
        )
        X_meta_buf = torch.empty(
            (batch_size, 4), dtype=torch.float32, device=self.device
        )

        # CPU output buffers for one chunk, reused across chunks
        effective_chunk_size = min(chunk_size, n_precursors)
        chunk_row_capacity = effective_chunk_size * max_fragments
        out_precursor_index_arr = np.empty(chunk_row_capacity, dtype=np.uint32)
        out_clevage_index_arr = np.empty(chunk_row_capacity, dtype=np.uint8)
        out_is_prefix_arr = np.empty(chunk_row_capacity, dtype=np.bool_)
        out_charge_arr = np.empty(chunk_row_capacity, dtype=np.uint8)
        out_is_modloss_arr = np.empty(chunk_row_capacity, dtype=np.bool_)
        out_num_rows_arr = np.empty(effective_chunk_size, dtype=np.uint8)
        all_counts = np.zeros(n_precursors, dtype=np.uint8)
        rows_written = 0
        out_mz_arr = np.empty(chunk_row_capacity, dtype=np.float32)
        out_pred_intensity_arr = np.empty(chunk_row_capacity, dtype=np.float32)
        out_rank_arr = np.empty(chunk_row_capacity, dtype=np.uint8)

        ms2_predictor = self.ms2_predictor

        def flush_chunk(chunk_start, chunk_end, processed_mask):
            if not processed_mask.all():
                missing = np.flatnonzero(~processed_mask) + chunk_start
                raise ValueError(
                    f"missing precursor(s) in chunk [{chunk_start}, {chunk_end}): "
                    f"{missing[:10]}"
                )
            nonlocal rows_written
            counts = out_num_rows_arr[: chunk_end - chunk_start]
            fragment_offsets(counts, start=rows_written)
            all_counts[chunk_start:chunk_end] = counts
            chunk_df = self._compact_speclib(
                {
                    "precursor_index": out_precursor_index_arr,
                    "cleavage_index": out_clevage_index_arr,
                    "is_prefix": out_is_prefix_arr,
                    "charge": out_charge_arr,
                    "is_modloss": out_is_modloss_arr,
                    "mz": out_mz_arr,
                    "predicted_intensity": out_pred_intensity_arr,
                    "rank": out_rank_arr,
                },
                counts,
                max_fragments,
            )
            if chunk_df.height:
                writer.write_table(chunk_df.to_arrow(), row_group_size=chunk_df.height)
            rows_written += chunk_df.height

        current_chunk_start = None
        current_chunk_end = None
        processed_mask = None

        with torch.inference_mode():
            for batch in Prefetcher(dl, transform=pin_tensor_dict):
                batch_precursor_index_arr = (
                    batch["precursor_index"].to(torch.uint32).numpy()
                )

                # the first row's chunk is the whole batch's
                batch_chunk_start = (
                    int(batch_precursor_index_arr[0]) // chunk_size
                ) * chunk_size
                if current_chunk_start != batch_chunk_start:
                    if current_chunk_start is not None:
                        flush_chunk(
                            current_chunk_start, current_chunk_end, processed_mask
                        )
                    current_chunk_start = batch_chunk_start
                    current_chunk_end = min(
                        current_chunk_start + chunk_size, n_precursors
                    )
                    processed_mask = np.zeros(
                        current_chunk_end - current_chunk_start, dtype=np.bool_
                    )

                # bounds check before the njit helper writes by local index
                if (
                    batch_precursor_index_arr.min() < current_chunk_start
                    or batch_precursor_index_arr.max() >= current_chunk_end
                ):
                    raise ValueError(
                        "batch precursor_index out of chunk range "
                        f"[{current_chunk_start}, {current_chunk_end})"
                    )

                local_idx = (
                    batch_precursor_index_arr.astype(np.int64) - current_chunk_start
                )
                if processed_mask[local_idx].any():
                    dup = batch_precursor_index_arr[processed_mask[local_idx]]
                    raise ValueError(f"duplicate precursor_index in chunk: {dup[:10]}")
                processed_mask[local_idx] = True

                x_aa_t = batch["x_aa"]
                x_mod_t = batch["x_mod"]
                x_meta_t = batch["x_meta"]

                n, L = x_aa_t.shape
                X_aa_buf[:n, :L].copy_(x_aa_t, non_blocking=True)
                X_mod_buf[:n, :L, :].copy_(x_mod_t, non_blocking=True)
                X_meta_buf[:n].copy_(x_meta_t, non_blocking=True)

                y_pred = ms2_predictor(
                    X_aa_buf[:n, :L], X_mod_buf[:n, :L, :], X_meta_buf[:n]
                )
                batch_intensity_arr = y_pred.detach().cpu().numpy()

                update_speclib_chunk_arr(
                    out_precursor_index_arr,
                    out_clevage_index_arr,
                    out_is_prefix_arr,
                    out_charge_arr,
                    out_is_modloss_arr,
                    out_mz_arr,
                    out_pred_intensity_arr,
                    out_rank_arr,
                    out_num_rows_arr,
                    prefix_mass_container,
                    ion_type_container,
                    peptidoform_index_arr,
                    batch_precursor_index_arr,
                    batch_intensity_arr,
                    chunk_start_precursor_index=current_chunk_start,
                    max_fragments=max_fragments,
                    min_intensity=self.min_intensity,
                    detectable_min_mz=detectable_min_mz,
                    detectable_max_mz=detectable_max_mz,
                )
                progress.advance(n)

            if current_chunk_start is not None:
                flush_chunk(current_chunk_start, current_chunk_end, processed_mask)
        return all_counts

    @staticmethod
    def _assert_contiguous_precursor_index(
        precursor_chunk_df: pl.DataFrame, chunk_start: int, chunk_end: int
    ):
        actual = precursor_chunk_df["precursor_index"].to_numpy()
        expected = np.arange(chunk_start, chunk_end, dtype=actual.dtype)
        if not np.array_equal(actual, expected):
            raise ValueError(
                "precursor_df.precursor_index must be contiguous and match row "
                f"position; chunk [{chunk_start}, {chunk_end}) violates this"
            )

    def predict_rt(
        self,
        peptide_df: pl.DataFrame,
        modification_df: pl.DataFrame,
        precursor_df: pl.DataFrame,
        batch_size: int = 512,
        progress: ProgressTracker = None,
    ):
        precursor_ds = PeptideDataset(
            precursor_df, modification_df, peptide_df, level="peptidoform"
        )
        batch_sampler = SeqDataBatchSampler(
            precursor_ds,
            batch_grouping_column="sequence_length",
            batch_size=batch_size,
            shuffle=False,
        )

        # a caller's tracker has an arbitrary total; drive a child sized to the
        # batch count instead
        total_batches = max(batch_sampler.count_num_of_batches(), 1)
        own_progress = progress is None
        if own_progress:
            progress = TqdmProgressTracker(
                total=total_batches, description="Predicting RT"
            )
            batch_progress = progress
        else:
            batch_progress = progress.create_child(
                "Predicting RT", total=total_batches, portion=100
            )

        dl = DataLoader(
            dataset=precursor_ds, batch_sampler=batch_sampler, num_workers=0
        )

        # GPU buffers reused for every H2D copy
        max_token_len = peptide_df["sequence_length"].max() + 3
        mod_feat_dim = self.rt_predictor.mod_embedding.in_features
        X_aa_buf = torch.empty(
            (batch_size, max_token_len), dtype=torch.long, device=self.device
        )
        X_mod_buf = torch.empty(
            (batch_size, max_token_len, mod_feat_dim),
            dtype=torch.float32,
            device=self.device,
        )

        out_peptidoform_index = np.empty(modification_df.shape[0], dtype=np.uint32)
        out_ref_rt = np.empty(modification_df.shape[0], dtype=np.float32)
        offset = 0
        rt_predictor = self.rt_predictor

        with torch.inference_mode():
            for batch in Prefetcher(dl, transform=pin_tensor_dict):
                peptidoform_index_arr = (
                    batch["peptidoform_index"].to(torch.uint32).numpy()
                )
                x_aa_t = batch["x_aa"]
                x_mod_t = batch["x_mod"]

                n, L = x_aa_t.shape
                X_aa_buf[:n, :L].copy_(x_aa_t, non_blocking=True)
                X_mod_buf[:n, :L, :].copy_(x_mod_t, non_blocking=True)

                y_pred = rt_predictor(X_aa_buf[:n, :L], X_mod_buf[:n, :L, :])

                out_peptidoform_index[offset : offset + n] = peptidoform_index_arr
                out_ref_rt[offset : offset + n] = (
                    y_pred.flatten().detach().cpu().numpy()
                )
                offset += n
                batch_progress.advance(1)

        if own_progress:
            progress.complete()
            progress.close()
        else:
            batch_progress.complete()

        return pl.DataFrame(
            {
                "peptidoform_index": out_peptidoform_index[:offset],
                "ref_rt": out_ref_rt[:offset],
            }
        )

    def generate_spectral_lib(
        self,
        peptide_df: pl.DataFrame,
        modification_df: pl.DataFrame,
        precursor_df: pl.DataFrame,
        prefix_mass_container: PrefixMassArrayContainer,
        progress: ProgressTracker = None,
        save_dir: Union[str, Path] = None,
        precursor_chunk_size: int = None,
        batch_size: int = 512,
    ):
        """Return (MS2 result, modification_df, precursor_df with fragment indexing).

        The MS2 result is the speclib DataFrame, or, when precursor_chunk_size is
        set, the Path of save_dir/speclib_df.parquet written in chunks of that
        many precursors. modification_df gains ref_rt.
        """
        if precursor_chunk_size is not None and save_dir is None:
            raise ValueError("save_dir is required when precursor_chunk_size is set")

        own_progress = progress is None
        if progress is None:
            progress = TqdmProgressTracker(
                total=100, description="Generating spectral library"
            )

        try:
            # MS2:RT progress is 80:20. total=100 is nominal: each predictor
            # drives its own sized child, so these are only completed.
            ms2_progress = progress.create_child(
                "Predicting MS2 spectra", total=100, portion=80
            )
            if precursor_chunk_size is None:
                ms2_result, precursor_df = self.predict_ms2_spectra(
                    peptide_df=peptide_df,
                    modification_df=modification_df,
                    precursor_df=precursor_df,
                    prefix_mass_container=prefix_mass_container,
                    batch_size=batch_size,
                    detectable_min_mz=self.min_fragment_mz,
                    detectable_max_mz=self.max_fragment_mz,
                    progress=ms2_progress,
                )
            else:
                ms2_result, precursor_df = self.predict_ms2_spectra_to_parquet(
                    peptide_df=peptide_df,
                    modification_df=modification_df,
                    precursor_df=precursor_df,
                    prefix_mass_container=prefix_mass_container,
                    save_dir=save_dir,
                    batch_size=batch_size,
                    chunk_size=precursor_chunk_size,
                    detectable_min_mz=self.min_fragment_mz,
                    detectable_max_mz=self.max_fragment_mz,
                    progress=ms2_progress,
                )
            ms2_progress.complete()

            rt_progress = progress.create_child("Predicting RT", total=100, portion=20)
            rt_df = self.predict_rt(
                peptide_df=peptide_df,
                modification_df=modification_df,
                precursor_df=precursor_df,
                batch_size=batch_size,
                progress=rt_progress,
            )
            rt_progress.complete()

            modification_df = modification_df.join(
                rt_df, on="peptidoform_index", how="left"
            )

            if own_progress:
                progress.complete()
        finally:
            if own_progress:
                progress.close()

        return ms2_result, modification_df, precursor_df
