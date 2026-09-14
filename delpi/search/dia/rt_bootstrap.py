"""RT-stratified, checkpointed, robust initial DIA bootstrap calibration.

Replaces the match-count-based `run_quick_search` stopping rule for the
*initial* DIA calibration step (before the first full search) with:

    RT-stratified candidate sampling (CandidatePool)
        -> quick matching in incremental checkpoints
        -> cumulative target-decoy analysis (AnchorSelector)
        -> anchor count / RT coverage check
        -> robust (MAD, degree<=3) calibration (BootstrapRTCalibrator)
        -> stop on success, else keep sampling within a hard budget
        -> if the budget is exhausted, a best-effort fit with whatever
           qualified anchors were collected (relaxed anchor-count floor)
        -> broad LinearProjectionCalibrator fallback only if even that
           best-effort fit fails

Orchestrated by `DIARTBootstrapCalibrator`. Only used by
`DIASearchEngine.perform_rt_bootstrap()`; DDA and all other RT calibration
stages (post-search, second-pass) are untouched.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import polars as pl

from delpi.database.spec_lib_reader import SpectralLibReader
from delpi.model.rt_calibrator import BootstrapRTCalibrator, LinearProjectionCalibrator
from delpi.search.dia.quick_search import _quick_match
from delpi.utils.fdr import calculate_q_value

logger = logging.getLogger(__name__)

_MATCH_SCHEMA = {
    "precursor_index": pl.UInt32,
    "peptidoform_index": pl.UInt32,
    "ref_rt": pl.Float32,
    "observed_rt": pl.Float32,
    "score": pl.Float32,
    "is_decoy": pl.Boolean,
}


@dataclass
class RTBootstrapConfig:
    n_bins: int = 10
    seed: int = 1226
    first_checkpoint: int = 20_000
    hard_limit: int = 1_000_000
    q_value_cutoff: float = 0.05
    min_unique_anchors: int = 300
    min_anchors_best_effort: int = 150
    min_anchors_per_bin: int = 5
    min_bin_coverage_frac: float = 0.8
    max_fit_anchors: int = 5_000
    mad_clip_thresh: float = 3.5
    max_mad_iters: int = 3
    max_degree: int = 3
    min_rt_tolerance: float = 0.10
    narrow_max_half_width_frac: float = 0.15
    broad_rt_tolerance: float = 0.30
    top_n_best_effort: int = 500
    max_fragments: int = 6
    max_precursor_isotopes: int = 1
    max_fragment_isotopes: int = 1

    def checkpoints(self) -> List[int]:
        cps, c = [], self.first_checkpoint
        while c < self.hard_limit:
            cps.append(c)
            c *= 2
        cps.append(self.hard_limit)
        return cps


@dataclass
class RTBootstrapResult:
    calibrator: object  # exposes .predict(ref_rt) -> df[predicted_rt, rt_lb, rt_ub]
    success: bool
    fallback_reason: Optional[str]
    n_rounds: int = 0
    n_evaluated: int = 0
    n_unique_anchors: int = 0
    diagnostics: dict = field(default_factory=dict)


class CandidatePool:
    """RT-stratified, target/decoy-balanced, without-replacement sampling pool.

    Each bin's target/decoy candidates are shuffled once (fixed seed);
    `take_round()` advances monotonic per-bin cursors, so a candidate is
    never returned twice and sampling is fully deterministic.
    """

    def __init__(
        self,
        precursor_index_arr: np.ndarray,
        bin_id_arr: np.ndarray,
        is_decoy_arr: np.ndarray,
        n_bins: int,
        seed: int,
    ):
        rng = np.random.RandomState(seed)
        self.n_bins = n_bins
        self._targets: List[np.ndarray] = []
        self._decoys: List[np.ndarray] = []
        self._target_cursor = [0] * n_bins
        self._decoy_cursor = [0] * n_bins
        bin_counts = np.zeros(n_bins, dtype=np.int64)
        for b in range(n_bins):
            in_bin = bin_id_arr == b
            t_idx = precursor_index_arr[in_bin & ~is_decoy_arr]
            d_idx = precursor_index_arr[in_bin & is_decoy_arr]
            self._targets.append(rng.permutation(t_idx))
            self._decoys.append(rng.permutation(d_idx))
            bin_counts[b] = t_idx.shape[0] + d_idx.shape[0]
        self.initial_nonempty_bins = int(np.sum(bin_counts > 0))

    def _remaining(self, b: int) -> int:
        return (
            self._targets[b].shape[0]
            - self._target_cursor[b]
            + self._decoys[b].shape[0]
            - self._decoy_cursor[b]
        )

    def total_remaining(self) -> int:
        return sum(self._remaining(b) for b in range(self.n_bins))

    def _take_from_bin(self, b: int, k: int) -> np.ndarray:
        t_rem = self._targets[b].shape[0] - self._target_cursor[b]
        d_rem = self._decoys[b].shape[0] - self._decoy_cursor[b]
        k = min(k, t_rem + d_rem)
        k_t = min(t_rem, (k + 1) // 2)
        k_d = min(d_rem, k - k_t)
        k_t = min(t_rem, k - k_d)  # redistribute back to target if decoy pool was short

        chunks = []
        for pool, cursor, count in (
            (self._targets, self._target_cursor, k_t),
            (self._decoys, self._decoy_cursor, k_d),
        ):
            if count > 0:
                s = cursor[b]
                cursor[b] += count
                chunks.append(pool[b][s : s + count])
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    def take_round(self, round_budget: int) -> np.ndarray:
        """Sample up to `round_budget` new candidates, spread as evenly as
        possible across still-active bins, redistributing any quota an
        exhausted bin couldn't use."""
        if round_budget <= 0:
            return np.empty(0, dtype=np.int64)

        selected, remaining_budget = [], round_budget
        active = [b for b in range(self.n_bins) if self._remaining(b) > 0]
        while remaining_budget > 0 and active:
            share = max(1, remaining_budget // len(active))
            progressed = False
            for b in list(active):
                if remaining_budget <= 0:
                    break
                picked = self._take_from_bin(
                    b, min(share, self._remaining(b), remaining_budget)
                )
                if picked.shape[0] > 0:
                    selected.append(picked)
                    remaining_budget -= picked.shape[0]
                    progressed = True
            active = [b for b in active if self._remaining(b) > 0]
            if not progressed:
                break
        return np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)


class AnchorSelector:
    """Turns cumulative quick-match results into deduplicated, q-value
    filtered RT anchors, and judges whether they cover the RT range well
    enough to attempt a calibration fit."""

    def __init__(self, bin_edges: np.ndarray, cfg: RTBootstrapConfig):
        self.bin_edges = bin_edges
        self.cfg = cfg

    def assign_bins(self, values: np.ndarray) -> np.ndarray:
        bin_id = np.digitize(values, self.bin_edges) - 1
        return np.clip(bin_id, 0, self.cfg.n_bins - 1)

    def select(self, cumulative_df: pl.DataFrame) -> Tuple[pl.DataFrame, np.ndarray]:
        """q-values from ALL accumulated target+decoy matches, then target
        filtering (q<=cutoff), finite-RT filtering, and one anchor per
        peptidoform (highest score, ties broken by precursor_index). No
        top-N fallback: an undersized result here means "keep collecting",
        not "relax the threshold"."""
        empty_counts = np.zeros(self.cfg.n_bins, dtype=np.int64)
        if cumulative_df.height == 0:
            return cumulative_df, empty_counts

        scored_df = calculate_q_value(
            cumulative_df, score_column="score", out_column="q_value"
        )
        target_df = scored_df.filter(
            (~pl.col("is_decoy"))
            & (pl.col("q_value") <= self.cfg.q_value_cutoff)
            & pl.col("ref_rt").is_finite()
            & pl.col("observed_rt").is_finite()
        )
        if target_df.height == 0:
            return target_df, empty_counts

        anchors_df = target_df.sort(
            ["score", "precursor_index"], descending=[True, False]
        ).unique("peptidoform_index", keep="first")

        bin_id = self.assign_bins(anchors_df["ref_rt"].to_numpy())
        anchors_df = anchors_df.with_columns(
            pl.Series("bin_id", bin_id, dtype=pl.Int32)
        )
        return anchors_df, np.bincount(bin_id, minlength=self.cfg.n_bins)

    def is_sufficient(
        self,
        n_unique_anchors: int,
        counts: np.ndarray,
        initial_nonempty_bins: int,
        min_unique_anchors: Optional[int] = None,
    ) -> bool:
        threshold = (
            self.cfg.min_unique_anchors
            if min_unique_anchors is None
            else min_unique_anchors
        )
        if n_unique_anchors < threshold:
            return False
        covered = int(np.sum(counts >= self.cfg.min_anchors_per_bin))
        return (
            covered / max(initial_nonempty_bins, 1)
        ) >= self.cfg.min_bin_coverage_frac

    def subsample_for_fit(self, anchors_df: pl.DataFrame) -> pl.DataFrame:
        """Deterministic, RT-stratified subsample (never top-scoring-only),
        capped at `cfg.max_fit_anchors`."""
        if anchors_df.height <= self.cfg.max_fit_anchors:
            return anchors_df
        rng = np.random.RandomState(self.cfg.seed)
        bin_ids = anchors_df["bin_id"].to_numpy()
        nonempty_bins = np.unique(bin_ids)
        quota = max(1, self.cfg.max_fit_anchors // max(len(nonempty_bins), 1))
        keep = [
            rng.choice(idx, size=quota, replace=False) if idx.shape[0] > quota else idx
            for idx in (np.flatnonzero(bin_ids == b) for b in nonempty_bins)
        ]
        keep_idx = np.sort(np.concatenate(keep))
        if keep_idx.shape[0] > self.cfg.max_fit_anchors:
            keep_idx = np.sort(
                rng.choice(keep_idx, size=self.cfg.max_fit_anchors, replace=False)
            )
        return anchors_df[keep_idx.tolist()]


class DIARTBootstrapCalibrator:
    """Orchestrates the initial DIA RT bootstrap: builds the RT-stratified
    candidate pool once, then alternates between incremental quick-match
    collection rounds and cumulative-anchor trial fits until a robust
    calibration passes QC, the hard workload budget is exhausted, or every
    eligible candidate has been searched.

    `spec_reader`/`match_fn` are optional injection points for testing;
    production callers should leave them as None.
    """

    def __init__(
        self,
        run,
        db_dir,
        ms2_tol_in_ppm: float = 10.0,
        cfg: Optional[RTBootstrapConfig] = None,
        figure_path: Optional[Path] = None,
        spec_reader: Optional[SpectralLibReader] = None,
        match_fn: Optional[
            Callable[[List[Tuple[int, int, int]], np.ndarray], pl.DataFrame]
        ] = None,
    ):
        self._dia_run = run
        self.ms2_tol_in_ppm = ms2_tol_in_ppm
        self.cfg = cfg or RTBootstrapConfig()
        self.figure_path = figure_path
        self._match_fn = match_fn or self._match_selected
        self.spec_reader = spec_reader or SpectralLibReader(
            peptide_db_path=db_dir,
            max_fragments=self.cfg.max_fragments,
            max_precursor_isotopes=self.cfg.max_precursor_isotopes,
            max_fragment_isotopes=self.cfg.max_fragment_isotopes,
        )
        self.min_rt_in_seconds = 0.0
        self.max_rt_in_seconds = run.gradient_length_in_seconds
        self._calibrator_kwargs = dict(
            min_rt_in_seconds=self.min_rt_in_seconds,
            max_rt_in_seconds=self.max_rt_in_seconds,
            max_degree=self.cfg.max_degree,
            mad_clip_thresh=self.cfg.mad_clip_thresh,
            max_mad_iters=self.cfg.max_mad_iters,
            min_rt_tolerance=self.cfg.min_rt_tolerance,
            max_rt_tolerance=self.cfg.narrow_max_half_width_frac,
        )
        self.bin_edges: Optional[np.ndarray] = None
        self.anchor_selector: Optional[AnchorSelector] = None

    # -- one-time setup ------------------------------------------------------

    def _broad_fallback_calibrator(self) -> LinearProjectionCalibrator:
        calibrator = LinearProjectionCalibrator(
            min_rt_in_seconds=self.min_rt_in_seconds,
            max_rt_in_seconds=self.max_rt_in_seconds,
            rt_tolerance=self.cfg.broad_rt_tolerance,
        ).fit(self.spec_reader.modification_df["ref_rt"])
        self.spec_reader.calibrate_rt(calibrator)
        return calibrator

    def _window_ranges(self) -> List[Tuple[int, int, int]]:
        """Disjoint (window_idx, min_precursor_index, max_precursor_index)
        tiles, one per DIA window, covering every precursor reachable by
        some window. Assumes `all_precursor_df` is sorted by precursor_mz
        (an existing invariant relied on elsewhere for the same reason);
        overlap between adjacent windows is resolved in favor of the
        lower-indexed (by min_idx) window."""
        raw_ranges = []
        for win_idx in range(self._dia_run.dia_scheme_df.shape[0]):
            dia_win = self._dia_run.get_dia_window(win_idx)
            if dia_win is None:
                continue
            min_idx, max_idx = self.spec_reader._get_index_range_by_mz(
                *dia_win.isolation_mz_range
            )
            if min_idx is not None and max_idx is not None:
                raw_ranges.append((win_idx, int(min_idx), int(max_idx)))

        raw_ranges.sort(key=lambda r: r[1])
        tiled, prev_max = [], -1
        for win_idx, lo, hi in raw_ranges:
            lo = max(lo, prev_max + 1)
            if lo <= hi:
                tiled.append((win_idx, lo, hi))
                prev_max = hi
        return tiled

    def _make_bin_edges(self, ref_rt_eligible: np.ndarray) -> np.ndarray:
        lo, hi = float(np.min(ref_rt_eligible)), float(np.max(ref_rt_eligible))
        return np.linspace(lo, hi if hi > lo else lo + 1.0, self.cfg.n_bins + 1)

    def _build_candidate_pool(
        self, window_ranges: List[Tuple[int, int, int]]
    ) -> Optional[CandidatePool]:
        """One-time, whole-DB arrays: per-(global)-precursor ref_rt,
        is_decoy, and which DIA window it belongs to; builds the fixed RT
        bin edges/selector and the RT-stratified sampling pool from them."""
        peptidoform_index_arr = (
            self.spec_reader.all_precursor_df.select("peptidoform_index")
            .collect()
            .to_series()
            .to_numpy()
            .astype(np.int64)
        )
        mod_is_decoy_arr = self.spec_reader.peptide_df["is_decoy"].to_numpy()[
            self.spec_reader.modification_df["peptide_index"].to_numpy()
        ]
        ref_rt_arr = self.spec_reader.modification_df["ref_rt"].to_numpy()[
            peptidoform_index_arr
        ]
        is_decoy_arr = mod_is_decoy_arr[peptidoform_index_arr]

        window_id_arr = np.full(peptidoform_index_arr.shape[0], -1, dtype=np.int32)
        for win_idx, lo, hi in window_ranges:
            window_id_arr[lo : hi + 1] = win_idx

        eligible = np.flatnonzero((window_id_arr >= 0) & np.isfinite(ref_rt_arr))
        if eligible.shape[0] == 0:
            return None

        self.bin_edges = self._make_bin_edges(ref_rt_arr[eligible])
        self.anchor_selector = AnchorSelector(self.bin_edges, self.cfg)
        bin_id = self.anchor_selector.assign_bins(ref_rt_arr[eligible])
        return CandidatePool(
            eligible,
            bin_id,
            is_decoy_arr[eligible],
            n_bins=self.cfg.n_bins,
            seed=self.cfg.seed,
        )

    # -- per-round matching ----------------------------------------------------

    def _match_selected(
        self, window_ranges: List[Tuple[int, int, int]], selected_global_idx: np.ndarray
    ) -> pl.DataFrame:
        """Group selected global precursor indices by DIA window and
        quick-match only that subset within each window's container (built
        once per window per round -- no whole-library rescans, no XIC
        arrays retained)."""
        if selected_global_idx.shape[0] == 0:
            return pl.DataFrame(schema=_MATCH_SCHEMA)

        order = np.argsort(selected_global_idx)
        sorted_idx = selected_global_idx[order]
        rows = []

        for win_idx, lo, hi in window_ranges:
            left = np.searchsorted(sorted_idx, lo, side="left")
            right = np.searchsorted(sorted_idx, hi, side="right")
            if right <= left:
                continue
            window_global_idx = sorted_idx[left:right]

            dia_win = self._dia_run.get_dia_window(win_idx)
            container = self.spec_reader.read_by_index_range(lo, hi)
            local_idx = (window_global_idx - lo).astype(np.int64)

            frame_index_arr, score_arr, valid_arr = _quick_match(
                container,
                dia_win.get_peak_container(),
                dia_win.get_frame_num_map(),
                local_idx,
                fragment_mz_tol=self.ms2_tol_in_ppm,
            )
            if not np.any(valid_arr):
                continue

            peptidoform_index_arr = container.precursor_peptidoform_index_arr[
                local_idx[valid_arr]
            ]
            peptide_index_arr = container.mod_peptide_index_arr[peptidoform_index_arr]
            observed_rt_arr = dia_win.meta_df[frame_index_arr[valid_arr]][
                "time_in_seconds"
            ].to_numpy()

            rows.append(
                pl.DataFrame(
                    {
                        "precursor_index": window_global_idx[valid_arr].astype(
                            np.uint32
                        ),
                        "peptidoform_index": peptidoform_index_arr.astype(np.uint32),
                        "ref_rt": container.mod_ref_rt_arr[
                            peptidoform_index_arr
                        ].astype(np.float32),
                        "observed_rt": observed_rt_arr.astype(np.float32),
                        "score": score_arr[valid_arr].astype(np.float32),
                        "is_decoy": container.peptide_is_decoy_arr[
                            peptide_index_arr
                        ].astype(np.bool_),
                    },
                    schema=_MATCH_SCHEMA,
                )
            )

        return (
            pl.concat(rows, how="vertical")
            if rows
            else pl.DataFrame(schema=_MATCH_SCHEMA)
        )

    # -- trial fitting -----------------------------------------------------

    def _try_fit(
        self,
        anchors_df: pl.DataFrame,
        initial_nonempty_bins: int,
        scale_floor: float,
        min_unique_anchors: Optional[int] = None,
    ):
        fit_df = self.anchor_selector.subsample_for_fit(anchors_df)
        calibrator, diag = BootstrapRTCalibrator(**self._calibrator_kwargs).fit(
            fit_df["ref_rt"].to_numpy(),
            fit_df["observed_rt"].to_numpy(),
            scale_floor=scale_floor,
        )
        if calibrator is None:
            return None, diag, fit_df

        # re-check support/coverage against the ORIGINAL qualified anchor
        # pool's bins, not just whatever survived aggressive MAD clipping
        inlier_ref_rt = fit_df["ref_rt"].to_numpy()[diag["inlier_mask"]]
        inlier_counts = np.bincount(
            self.anchor_selector.assign_bins(inlier_ref_rt), minlength=self.cfg.n_bins
        )
        if not self.anchor_selector.is_sufficient(
            int(diag["inlier_mask"].sum()),
            inlier_counts,
            initial_nonempty_bins,
            min_unique_anchors=min_unique_anchors,
        ):
            diag["reason"] = "insufficient_support_after_outlier_rejection"
            return None, diag, fit_df
        return calibrator, diag, fit_df

    def _top_score_fit(
        self, cumulative_df: pl.DataFrame, scale_floor: float
    ) -> Tuple[Optional[BootstrapRTCalibrator], dict, pl.DataFrame]:
        """Absolute last resort before the broad fallback: ignore the
        q-value/RT-bin-coverage requirements entirely and fit directly on
        the ``top_n_best_effort`` highest-scoring target matches (one per
        peptidoform), still subject to `BootstrapRTCalibrator`'s own robust
        MAD/monotonicity/residual-width QC."""
        target_df = cumulative_df.filter(
            (~pl.col("is_decoy"))
            & pl.col("ref_rt").is_finite()
            & pl.col("observed_rt").is_finite()
        )
        if target_df.height == 0:
            return None, {"reason": "no_target_matches"}, target_df

        top_df = (
            target_df.sort(["score", "precursor_index"], descending=[True, False])
            .unique("peptidoform_index", keep="first")
            .head(self.cfg.top_n_best_effort)
        )
        calibrator, diag = BootstrapRTCalibrator(**self._calibrator_kwargs).fit(
            top_df["ref_rt"].to_numpy(),
            top_df["observed_rt"].to_numpy(),
            scale_floor=scale_floor,
        )
        return calibrator, diag, top_df

    def _save_diagnostic_figure(
        self,
        calibrator,
        anchors_df: pl.DataFrame,
        fit_df: pl.DataFrame,
        diagnostics: dict,
        success: bool,
    ) -> None:
        if self.figure_path is None:
            return

        try:
            from delpi.utils.plot import plot_rt_bootstrap_diagnostics, plt

            if self.bin_edges is None:
                ref_rt = self.spec_reader.modification_df["ref_rt"].to_numpy()
                ref_rt = ref_rt[np.isfinite(ref_rt)]
                bin_edges = (
                    self._make_bin_edges(ref_rt)
                    if ref_rt.size
                    else np.linspace(0.0, 1.0, self.cfg.n_bins + 1)
                )
            else:
                bin_edges = self.bin_edges

            fig = plot_rt_bootstrap_diagnostics(
                calibrator,
                anchors_df,
                fit_df,
                diagnostics,
                bin_edges,
                success,
            )
            self.figure_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.figure_path, dpi=150)
            plt.close(fig)
            logger.info(
                f"[{self._dia_run.name}] Saved DIA RT bootstrap diagnostics: "
                f"{self.figure_path}"
            )
        except Exception:
            logger.exception(
                f"[{self._dia_run.name}] Failed to save DIA RT bootstrap diagnostics"
            )

    # -- orchestration -----------------------------------------------------

    def run(self) -> RTBootstrapResult:
        broad_calibrator = self._broad_fallback_calibrator()
        empty_df = pl.DataFrame(schema=_MATCH_SCHEMA)
        window_ranges = self._window_ranges()
        if not window_ranges:
            return self._fallback_result(
                broad_calibrator,
                "no_dia_windows_mapped_to_library",
                anchors_df=empty_df,
                fit_df=empty_df,
            )

        pool = self._build_candidate_pool(window_ranges)
        if pool is None:
            return self._fallback_result(
                broad_calibrator,
                "no_eligible_candidates",
                anchors_df=empty_df,
                fit_df=empty_df,
            )

        scale_floor = max(2.0 * float(self._dia_run.cycle_time_in_seconds), 1.0)
        cumulative_rounds: List[pl.DataFrame] = []
        n_rounds, n_evaluated = 0, 0
        last_diag = {"reason": "no_rounds_executed"}
        anchors_df = empty_df
        fit_df = empty_df

        for checkpoint in self.cfg.checkpoints():
            round_budget = min(checkpoint, self.cfg.hard_limit) - n_evaluated
            if round_budget > 0 and pool.total_remaining() > 0:
                selected = pool.take_round(round_budget)
                if selected.shape[0] > 0:
                    cumulative_rounds.append(self._match_fn(window_ranges, selected))
                    n_evaluated += int(selected.shape[0])
                    n_rounds += 1

            if cumulative_rounds:
                cumulative_df = pl.concat(cumulative_rounds, how="vertical")
                anchors_df, counts = self.anchor_selector.select(cumulative_df)
                if self.anchor_selector.is_sufficient(
                    anchors_df.height, counts, pool.initial_nonempty_bins
                ):
                    calibrator, diag, fit_df = self._try_fit(
                        anchors_df, pool.initial_nonempty_bins, scale_floor
                    )
                    last_diag = diag
                    if calibrator is not None:
                        logger.info(
                            f"[{self._dia_run.name}] DIA RT bootstrap succeeded: rounds={n_rounds} "
                            f"evaluated={n_evaluated} anchors={anchors_df.height} "
                            f"degree={diag.get('degree')} half_width={diag.get('half_width', float('nan')):.1f}s"
                        )
                        result = RTBootstrapResult(
                            calibrator=calibrator,
                            success=True,
                            fallback_reason=None,
                            n_rounds=n_rounds,
                            n_evaluated=n_evaluated,
                            n_unique_anchors=anchors_df.height,
                            diagnostics=diag,
                        )
                        self._save_diagnostic_figure(
                            calibrator, anchors_df, fit_df, diag, success=True
                        )
                        return result

            if n_evaluated >= self.cfg.hard_limit or pool.total_remaining() <= 0:
                break

        n_unique_anchors = 0
        if cumulative_rounds:
            cumulative_df = pl.concat(cumulative_rounds, how="vertical")
            anchors_df, _counts = self.anchor_selector.select(cumulative_df)
            n_unique_anchors = anchors_df.height

            # Budget/candidates exhausted without ever reaching
            # min_unique_anchors: rather than jumping straight to the very
            # wide broad fallback bounds, make one last attempt with
            # whatever qualified anchors were actually collected (relaxed
            # anchor-count floor, same coverage/monotonicity/residual QC).
            if n_unique_anchors >= self.cfg.min_anchors_best_effort:
                calibrator, diag, fit_df = self._try_fit(
                    anchors_df,
                    pool.initial_nonempty_bins,
                    scale_floor,
                    min_unique_anchors=self.cfg.min_anchors_best_effort,
                )
                last_diag = diag
                if calibrator is not None:
                    logger.warning(
                        f"[{self._dia_run.name}] DIA RT bootstrap best-effort fit used "
                        f"(anchors={n_unique_anchors} below target={self.cfg.min_unique_anchors}): "
                        f"rounds={n_rounds} evaluated={n_evaluated} "
                        f"degree={diag.get('degree')} half_width={diag.get('half_width', float('nan')):.1f}s"
                    )
                    result = RTBootstrapResult(
                        calibrator=calibrator,
                        success=True,
                        fallback_reason=None,
                        n_rounds=n_rounds,
                        n_evaluated=n_evaluated,
                        n_unique_anchors=n_unique_anchors,
                        diagnostics=diag,
                    )
                    self._save_diagnostic_figure(
                        calibrator, anchors_df, fit_df, diag, success=True
                    )
                    return result

            # Still no usable fit (either too few q-value-qualified anchors,
            # or the best-effort fit above failed QC): drop the q-value/
            # RT-bin-coverage requirements entirely and fit on the raw
            # top-scoring target matches, before giving up to the broad
            # fallback bounds.
            calibrator, diag, fit_df = self._top_score_fit(cumulative_df, scale_floor)
            last_diag = diag
            if calibrator is not None:
                logger.warning(
                    f"[{self._dia_run.name}] DIA RT bootstrap top-score fallback fit used "
                    f"(anchors={n_unique_anchors} below target={self.cfg.min_unique_anchors}, "
                    f"top_n={fit_df.height}): rounds={n_rounds} evaluated={n_evaluated} "
                    f"degree={diag.get('degree')} half_width={diag.get('half_width', float('nan')):.1f}s"
                )
                result = RTBootstrapResult(
                    calibrator=calibrator,
                    success=True,
                    fallback_reason=None,
                    n_rounds=n_rounds,
                    n_evaluated=n_evaluated,
                    n_unique_anchors=fit_df.height,
                    diagnostics=diag,
                )
                self._save_diagnostic_figure(
                    calibrator, anchors_df, fit_df, diag, success=True
                )
                return result

        reason = last_diag.get(
            "reason", "budget_or_candidates_exhausted_without_valid_fit"
        )
        return self._fallback_result(
            broad_calibrator,
            reason,
            n_rounds,
            n_evaluated,
            n_unique_anchors,
            anchors_df,
            fit_df,
            last_diag,
        )

    def _fallback_result(
        self,
        calibrator,
        reason: str,
        n_rounds: int = 0,
        n_evaluated: int = 0,
        n_unique_anchors: int = 0,
        anchors_df: Optional[pl.DataFrame] = None,
        fit_df: Optional[pl.DataFrame] = None,
        diagnostics: Optional[dict] = None,
    ) -> RTBootstrapResult:
        logger.warning(
            f"[{self._dia_run.name}] DIA RT bootstrap failed, using broad fallback bounds: "
            f"reason={reason} rounds={n_rounds} evaluated={n_evaluated} anchors={n_unique_anchors}"
        )
        diagnostics = diagnostics or {"reason": reason}
        result = RTBootstrapResult(
            calibrator=calibrator,
            success=False,
            fallback_reason=reason,
            n_rounds=n_rounds,
            n_evaluated=n_evaluated,
            n_unique_anchors=n_unique_anchors,
            diagnostics=diagnostics,
        )
        empty_df = pl.DataFrame(schema=_MATCH_SCHEMA)
        self._save_diagnostic_figure(
            calibrator,
            anchors_df if anchors_df is not None else empty_df,
            fit_df if fit_df is not None else empty_df,
            diagnostics,
            success=False,
        )
        return result
