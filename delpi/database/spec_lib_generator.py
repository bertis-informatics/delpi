import os
import uuid
from pathlib import Path
from typing import Union

import polars as pl
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import torch
from torch.utils.data import DataLoader

from delpi.lcms.base_ion_type import BaseIonType
from delpi.lcms.fragmentation import Fragmentation
from delpi.lcms.neutral_loss import NeutralLoss
from delpi.model.spec_lib import Ms2SpectrumPredictor, RetentionTimePredictor
from delpi.model.spec_lib.dataset import PeptideDataset
from delpi.utils.batch_sampler import SeqDataBatchSampler, ChunkedSeqDataBatchSampler
from delpi.database.numba.prefix_mass_array import PrefixMassArrayContainer
from delpi.database.numba.spec_lib_utils import (
    update_speclib_arr,
    update_speclib_chunk_arr,
)
from delpi.utils.prefetch import Prefetcher, pin_tensor_dict
from delpi.search.progress import ProgressTracker, TqdmProgressTracker
from delpi import MODEL_DIR


class SpectralLibGenerator:

    speclib_df_schema = {
        "precursor_index": pl.UInt32,
        "cleavage_index": pl.UInt8,
        "is_prefix": pl.Boolean,
        "charge": pl.UInt8,
        "mz": pl.Float32,
        "predicted_intensity": pl.Float32,
        "rank": pl.UInt8,
    }

    def __init__(
        self,
        min_charge: int = 1,
        max_charge: int = 2,
        prefix_ion_type=BaseIonType.B,
        suffix_ion_type=BaseIonType.Y,
        max_fragments: int = 16,
        apply_phospho=False,
        device: Union[str, torch.device] = "cuda:0",
        ms2_predictor: Ms2SpectrumPredictor = None,
        rt_predictor: RetentionTimePredictor = None,
    ):
        self.max_fragments = max_fragments
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
        }

    def predict_ms2_spectra(
        self,
        peptide_df: pl.DataFrame,
        modification_df: pl.DataFrame,
        precursor_df: pl.DataFrame,
        prefix_mass_container: PrefixMassArrayContainer,
        batch_size: int = 512,
        detectable_min_mz: float = 200,
        detectable_max_mz: float = 1800,
        progress: ProgressTracker = None,
    ):
        precursor_ds = PeptideDataset(
            precursor_df, modification_df, peptide_df, level="precursor"
        )
        # include_modloss = len(self.fragmentation.neutral_losses) > 1
        batch_sampler = SeqDataBatchSampler(
            precursor_ds,
            batch_grouping_column="sequence_length",
            batch_size=batch_size,
            shuffle=False,
        )

        # Owns (creates + completes/closes) its tracker only when the caller
        # didn't supply one. An externally-supplied tracker's own total is
        # set arbitrarily by its creator, so a fresh child sized to the real
        # batch count is created here to drive the subtask bar (position 1);
        # the caller's tracker only receives proportional progress via that
        # child's advance()/complete() calls.
        total_batches = max(batch_sampler.count_num_of_batches(), 1)
        own_progress = progress is None
        if own_progress:
            progress = TqdmProgressTracker(
                total=total_batches, description="Predicting MS2 spectra"
            )
            batch_progress = progress
        else:
            batch_progress = progress.create_child(
                "Predicting MS2 spectra", total=total_batches, portion=100
            )

        dl = DataLoader(
            dataset=precursor_ds, batch_sampler=batch_sampler, num_workers=0
        )

        # Pre-allocate GPU buffers for zero-allocation H2D transfer
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

        ion_type_container = self.fragmentation.get_ion_types()
        peptidoform_index_arr = precursor_df["peptidoform_index"].to_numpy()
        max_fragments = self.max_fragments
        speclib_row_count = max_fragments * precursor_df.shape[0]

        out_precursor_index_arr = np.empty(speclib_row_count, dtype=np.uint32)
        out_clevage_index_arr = np.empty(speclib_row_count, dtype=np.uint8)
        out_is_prefix_arr = np.empty(speclib_row_count, dtype=np.bool_)
        out_charge_arr = np.empty(speclib_row_count, dtype=np.uint8)
        out_mz_arr = np.empty(speclib_row_count, dtype=np.float32)
        out_pred_intensity_arr = np.empty(speclib_row_count, dtype=np.float32)
        out_rank_arr = np.empty(speclib_row_count, dtype=np.uint8)
        ms2_predictor = self.ms2_predictor

        with torch.inference_mode():
            for batch in Prefetcher(dl, transform=pin_tensor_dict):
                batch_precursor_index_arr = (
                    batch["precursor_index"].to(torch.uint32).numpy()
                )
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

                update_speclib_arr(
                    out_precursor_index_arr,
                    out_clevage_index_arr,
                    out_is_prefix_arr,
                    out_charge_arr,
                    out_mz_arr,
                    out_pred_intensity_arr,
                    out_rank_arr,
                    prefix_mass_container,
                    ion_type_container,
                    peptidoform_index_arr,
                    batch_precursor_index_arr,
                    batch_intensity_arr,
                    max_fragments=max_fragments,
                    detectable_min_mz=detectable_min_mz,
                    detectable_max_mz=detectable_max_mz,
                )
                batch_progress.advance(1)

        if own_progress:
            progress.complete()
            progress.close()
        else:
            batch_progress.complete()

        return pl.DataFrame(
            {
                "precursor_index": out_precursor_index_arr,
                "cleavage_index": out_clevage_index_arr,
                "is_prefix": out_is_prefix_arr,
                "charge": out_charge_arr,
                "mz": out_mz_arr,
                "predicted_intensity": out_pred_intensity_arr,
                "rank": out_rank_arr,
            },
            schema=self.speclib_df_schema,
        )

    def predict_ms2_spectra_to_parquet(
        self,
        peptide_df: pl.DataFrame,
        modification_df: pl.DataFrame,
        precursor_df: pl.DataFrame,
        prefix_mass_container: PrefixMassArrayContainer,
        save_dir: Union[str, Path],
        batch_size: int = 1024,
        chunk_size: int = 65_536,
        detectable_min_mz: float = 200,
        detectable_max_mz: float = 1800,
        progress: ProgressTracker = None,
    ) -> Path:
        """Predict MS2 spectra and stream the result to a single Parquet file.

        Unlike predict_ms2_spectra(), this never allocates output arrays for
        the full library: only one chunk's worth of output rows
        (chunk_size * max_fragments) is held in memory at a time. The input
        pipeline (Dataset/Sampler/DataLoader/Prefetcher/GPU buffers) is
        still built only once for the whole prediction; chunk boundaries
        are detected from the batch stream itself.
        """

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        output_path = Path(save_dir) / "speclib_df.parquet"
        n_precursors = precursor_df.shape[0]

        # Built once from the schema so column order/dtype match speclib_df_schema.
        arrow_schema = pl.DataFrame(schema=self.speclib_df_schema).to_arrow().schema

        # Owned (created + completed/closed here) only when the caller didn't
        # supply a tracker; a single bar spans the whole task, not per-chunk.
        # An externally-supplied tracker's own total is set arbitrarily by
        # its creator, so a fresh child sized to the real precursor count is
        # created here to drive the subtask bar (position 1); the caller's
        # tracker only receives proportional progress via that child's
        # advance()/complete() calls. Progress unit is precursors processed,
        # not batches, since batch count would require a Dataset/Sampler
        # pre-pass this refactor is specifically meant to avoid.
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
        try:
            if n_precursors > 0:
                self._write_ms2_spectra_chunks(
                    writer,
                    arrow_schema,
                    peptide_df=peptide_df,
                    modification_df=modification_df,
                    precursor_df=precursor_df,
                    prefix_mass_container=prefix_mass_container,
                    batch_size=batch_size,
                    chunk_size=chunk_size,
                    detectable_min_mz=detectable_min_mz,
                    detectable_max_mz=detectable_max_mz,
                    progress=batch_progress,
                )
            writer.close()
            writer = None
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

        return output_path

    def _write_ms2_spectra_chunks(
        self,
        writer: pq.ParquetWriter,
        arrow_schema: pa.Schema,
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

        # precursor_index must be contiguous/global (row position == ID):
        # relied on below both for chunk-boundary detection from a batch's
        # precursor_index values, and by update_speclib_chunk_arr's local-
        # index math.
        self._assert_contiguous_precursor_index(precursor_df, 0, n_precursors)

        # global mapping: indexed by global precursor_index throughout.
        peptidoform_index_arr = precursor_df["peptidoform_index"].to_numpy()
        ion_type_container = self.fragmentation.get_ion_types()

        # Dataset/Sampler/DataLoader/Prefetcher are built exactly once for
        # the whole prediction; ChunkedSeqDataBatchSampler groups batches by
        # sequence_length while guaranteeing no batch spans two chunks, so
        # chunk transitions can be detected from the batch stream alone.
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

        # CPU output buffers are reused across chunks; sized once at the
        # largest possible chunk so memory stays proportional to chunk_size.
        effective_chunk_size = min(chunk_size, n_precursors)
        chunk_row_capacity = effective_chunk_size * max_fragments
        out_precursor_index_arr = np.empty(chunk_row_capacity, dtype=np.uint32)
        out_clevage_index_arr = np.empty(chunk_row_capacity, dtype=np.uint8)
        out_is_prefix_arr = np.empty(chunk_row_capacity, dtype=np.bool_)
        out_charge_arr = np.empty(chunk_row_capacity, dtype=np.uint8)
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
            chunk_row_count = (chunk_end - chunk_start) * max_fragments
            chunk_table = pa.Table.from_pydict(
                {
                    "precursor_index": out_precursor_index_arr[:chunk_row_count],
                    "cleavage_index": out_clevage_index_arr[:chunk_row_count],
                    "is_prefix": out_is_prefix_arr[:chunk_row_count],
                    "charge": out_charge_arr[:chunk_row_count],
                    "mz": out_mz_arr[:chunk_row_count],
                    "predicted_intensity": out_pred_intensity_arr[:chunk_row_count],
                    "rank": out_rank_arr[:chunk_row_count],
                },
                schema=arrow_schema,
            )
            writer.write_table(chunk_table, row_group_size=chunk_row_count)

        current_chunk_start = None
        current_chunk_end = None
        processed_mask = None

        with torch.inference_mode():
            for batch in Prefetcher(dl, transform=pin_tensor_dict):
                batch_precursor_index_arr = (
                    batch["precursor_index"].to(torch.uint32).numpy()
                )

                # Chunk boundaries are inferred from global precursor_index
                # values; the sampler guarantees a batch never spans two
                # chunks, so the first row's chunk is the whole batch's.
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

                # validate batch IDs are within the current chunk range
                # before handing them to the JIT helper.
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
                    out_mz_arr,
                    out_pred_intensity_arr,
                    out_rank_arr,
                    prefix_mass_container,
                    ion_type_container,
                    peptidoform_index_arr,
                    batch_precursor_index_arr,
                    batch_intensity_arr,
                    chunk_start_precursor_index=current_chunk_start,
                    max_fragments=max_fragments,
                    detectable_min_mz=detectable_min_mz,
                    detectable_max_mz=detectable_max_mz,
                )
                progress.advance(n)

            if current_chunk_start is not None:
                flush_chunk(current_chunk_start, current_chunk_end, processed_mask)

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

        # An externally-supplied tracker's own total is set arbitrarily by
        # its creator, so a fresh child sized to the real batch count is
        # created here to drive the subtask bar (position 1); the caller's
        # tracker only receives proportional progress via that child's
        # advance()/complete() calls.
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

        # Pre-allocate GPU buffers for zero-allocation H2D transfer
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

        # Pre-allocate output arrays
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
        min_fragment_mz: float = 200,
        max_fragment_mz: float = 1800,
        progress: ProgressTracker = None,
        save_dir: Union[str, Path] = None,
        precursor_chunk_size: int = None,
        batch_size: int = 512,
    ):
        """Predict RT and MS2 spectra and assemble the spectral library.

        When precursor_chunk_size is None (default), MS2 spectra are
        predicted in-memory via predict_ms2_spectra() and the first
        returned value is a pl.DataFrame. When precursor_chunk_size is
        given, MS2 spectra are instead streamed to ms2_output_path via
        predict_ms2_spectra_to_parquet() (at most precursor_chunk_size
        precursors held in memory at a time), and the first returned
        value is the Path to that Parquet file instead.
        """
        if precursor_chunk_size is not None and save_dir is None:
            raise ValueError("save_dir is required when precursor_chunk_size is set")

        own_progress = progress is None
        if progress is None:
            progress = TqdmProgressTracker(
                total=100, description="Generating spectral library"
            )

        try:
            # RT and MS2 prediction split the parent's progress ~1:4. total=100
            # here is a nominal placeholder: predict_rt()/predict_ms2_spectra()
            # re-create their own child (sized to the real batch count) from
            # whatever tracker they're given, so this total is never advanced
            # against directly, only completed as a whole.
            ms2_progress = progress.create_child(
                "Predicting MS2 spectra", total=100, portion=80
            )
            if precursor_chunk_size is None:
                ms2_result = self.predict_ms2_spectra(
                    peptide_df=peptide_df,
                    modification_df=modification_df,
                    precursor_df=precursor_df,
                    prefix_mass_container=prefix_mass_container,
                    batch_size=batch_size,
                    detectable_min_mz=min_fragment_mz,
                    detectable_max_mz=max_fragment_mz,
                    progress=ms2_progress,
                )
            else:
                ms2_result = self.predict_ms2_spectra_to_parquet(
                    peptide_df=peptide_df,
                    modification_df=modification_df,
                    precursor_df=precursor_df,
                    prefix_mass_container=prefix_mass_container,
                    save_dir=save_dir,
                    batch_size=batch_size,
                    chunk_size=precursor_chunk_size,
                    detectable_min_mz=min_fragment_mz,
                    detectable_max_mz=max_fragment_mz,
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

        return ms2_result, modification_df
