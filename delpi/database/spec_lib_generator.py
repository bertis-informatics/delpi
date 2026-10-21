import os
import uuid
from pathlib import Path
from typing import Sequence, Union

import polars as pl
import numpy as np
import pyarrow.parquet as pq

import torch
from torch.utils.data import DataLoader

from delpi.constants import DEFAULT_NCE
from delpi.lcms.base_ion_type import BaseIonType
from delpi.lcms.fragmentation import Fragmentation, get_frag_type_str
from delpi.lcms.neutral_loss import NeutralLoss
from delpi.lcms.data_container import IonTypeContainer
from delpi.constants import COMMON_NEUTRAL_LOSS_INDEX
from delpi.database.spec_lib_schema import (
    SPECLIB_SCHEMA,
    NEUTRAL_LOSS_DTYPE,
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
    """Generate libraries with intensity / family_rank**gamma fragment selection.

    Families share (cleavage_index, is_prefix) across charges and neutral losses.
    min_intensity is the inclusive normalized-intensity threshold; gamma=0
    recovers intensity-only ranking. max_fragments is an upper bound.
    Selected fragments are stored in increasing selection-score order, with
    rank 1 assigned to the fragment with the highest score.
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
        neutral_losses: Sequence[str | NeutralLoss] = None,
        min_intensity: float = 0.01,
        device: Union[str, torch.device] = "cuda:0",
        ms2_predictor: Ms2SpectrumPredictor = None,
        rt_predictor: RetentionTimePredictor = None,
        gamma: float = 0.5,
        nce: float = DEFAULT_NCE,
    ):
        if not isinstance(max_fragments, int) or not 1 <= max_fragments <= 255:
            raise ValueError("max_fragments must be an integer in [1, 255].")
        if not 0 <= min_intensity <= 1:
            raise ValueError("min_intensity must be in [0, 1].")
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative.")
        if not 1 <= min_charge <= max_charge <= 2:
            raise ValueError("The MS2 predictor supports fragment charges 1 and 2.")
        if not 0 <= min_fragment_mz <= max_fragment_mz:
            raise ValueError("Invalid fragment m/z range.")
        if not np.isfinite(nce) or nce < 0:
            raise ValueError("NCE must be finite and nonnegative")
        self.nce = float(nce)
        self.max_fragments = max_fragments
        self.min_intensity = min_intensity
        self.gamma = float(gamma)
        self.min_fragment_mz = min_fragment_mz
        self.max_fragment_mz = max_fragment_mz
        if neutral_losses is None:
            neutral_losses = NeutralLoss.get_common_neutral_losses()
        if isinstance(neutral_losses, str):
            raise ValueError("neutral_losses must be a sequence of loss symbols.")
        self.neutral_losses = tuple(
            loss.symbol if isinstance(loss, NeutralLoss) else loss
            for loss in neutral_losses
        )
        if (
            not self.neutral_losses
            or any(
                loss not in COMMON_NEUTRAL_LOSS_INDEX for loss in self.neutral_losses
            )
            or len(set(self.neutral_losses)) != len(self.neutral_losses)
        ):
            raise ValueError("neutral_losses must contain unique common loss symbols.")

        self.fragmentation = Fragmentation(
            min_charge,
            max_charge,
            prefix_ion_type=prefix_ion_type,
            suffix_ion_type=suffix_ion_type,
            max_fragment_isotopes=2,
            neutral_losses=[NeutralLoss.get(loss) for loss in self.neutral_losses],
        )

        self.device = device
        self._ms2_predictor = ms2_predictor
        self._rt_predictor = rt_predictor
        self._ion_type_container, self._model_columns = self._get_prediction_layout()

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
            "neutral_losses": list(self.neutral_losses),
            "min_intensity": self.min_intensity,
            "gamma": self.gamma,
            "nce": self.nce,
            "format_version": 2,
            "min_mz": self.min_fragment_mz,
            "max_mz": self.max_fragment_mz,
        }

    def _get_prediction_layout(self):
        """Resolve requested ions to model output columns during initialization."""
        table = self.fragmentation.ion_type_df.filter(pl.col("isotope_index") == 0)
        model_columns = {
            name: i for i, name in enumerate(self.ms2_predictor.fragment_types)
        }
        columns = []
        for row in table.iter_rows(named=True):
            loss = NeutralLoss.get(row["neutral_loss"])
            name = get_frag_type_str(
                BaseIonType.get(row["base_ion_type"]), loss, row["charge"]
            )
            if name not in model_columns:
                raise ValueError(
                    f"MS2 model does not predict requested fragment {name!r}."
                )
            columns.append(model_columns[name])
        ion_types = IonTypeContainer(
            table["is_prefix"].to_numpy(),
            table["charge"].to_numpy(),
            table["offset_mass"].to_numpy(),
            table["neutral_loss"].to_physical().cast(pl.UInt8).to_numpy(),
        )
        return ion_types, np.asarray(columns)

    @staticmethod
    def _compact_speclib(arrays, counts, max_fragments):
        capacity = len(counts) * max_fragments
        keep = (np.arange(max_fragments)[None, :] < counts[:, None]).ravel()
        data = {name: values[:capacity][keep] for name, values in arrays.items()}
        data["neutral_loss"] = pl.Series(
            "neutral_loss", data["neutral_loss"], dtype=pl.UInt8
        ).cast(NEUTRAL_LOSS_DTYPE)
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
        """Write MS2 Parquet and return (path, precursor_df with row indexing).

        The caller saves the returned precursor_df with the other database
        tables, e.g. via PeptideDatabase.save().

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

        Path(save_dir).mkdir(parents=True, exist_ok=True)
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

        # precursor_index must be contiguous/global (row position == ID):
        # relied on below both for chunk-boundary detection from a batch's
        # precursor_index values, and by update_speclib_chunk_arr's local-
        # index math.
        self._assert_contiguous_precursor_index(precursor_df, 0, n_precursors)

        # global mapping: indexed by global precursor_index throughout.
        peptidoform_index_arr = precursor_df["peptidoform_index"].to_numpy()
        ion_type_container = self._ion_type_container
        model_columns = self._model_columns

        # Dataset/Sampler/DataLoader/Prefetcher are built exactly once for
        # the whole prediction; ChunkedSeqDataBatchSampler groups batches by
        # sequence_length while guaranteeing no batch spans two chunks, so
        # chunk transitions can be detected from the batch stream alone.
        precursor_ds = PeptideDataset(
            precursor_df, modification_df, peptide_df, level="precursor", nce=self.nce
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
        out_neutral_loss_arr = np.empty(chunk_row_capacity, dtype=np.uint8)
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
                    "neutral_loss": out_neutral_loss_arr,
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

        pin = pin_tensor_dict if torch.device(self.device).type == "cuda" else None
        with torch.inference_mode():
            for batch in Prefetcher(dl, transform=pin):
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
                batch_intensity_arr = np.ascontiguousarray(
                    y_pred.detach().cpu().numpy()[..., model_columns]
                )

                update_speclib_chunk_arr(
                    out_precursor_index_arr,
                    out_clevage_index_arr,
                    out_is_prefix_arr,
                    out_charge_arr,
                    out_neutral_loss_arr,
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
                    gamma=self.gamma,
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

        pin = pin_tensor_dict if torch.device(self.device).type == "cuda" else None
        with torch.inference_mode():
            for batch in Prefetcher(dl, transform=pin):
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
