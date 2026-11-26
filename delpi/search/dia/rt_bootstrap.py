"""Initial DIA RT calibration by incremental quick matching.

Used by `DIASearchEngine.perform_rt_bootstrap()` before the first full search,
in place of `run_quick_search`'s match-count stopping rule. DIA windows are
quick-matched one at a time in random order, all precursors of a window at once.
After each window, the q-value-filtered anchors from all matches so far
(`AnchorSelector`) are counted; once there are enough, a robust Huber fit of
degree <= 2 (`BootstrapRTCalibrator`) is tried, and the first fit that passes
QC is returned. Matching stops at a workload budget (`hard_limit` precursors).
If the budget or the windows run out, one fit with the relaxed
`min_anchors_best_effort` floor is tried before falling back to broad
`LinearProjectionCalibrator` bounds.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

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

# bins of the anchor-coverage panel in the diagnostic figure only
_PLOT_N_BINS = 10


@dataclass
class RTBootstrapConfig:
    seed: int = 1226
    hard_limit: int = 10_000_000
    min_window_frac: float = 0.25
    q_value_cutoff: float = 0.05
    min_unique_anchors: int = 300
    min_anchors_best_effort: int = 150
    top_n_best_effort: int = 300
    max_fit_anchors: int = 5_000
    max_degree: int = 2
    min_rt_tolerance: float = 0.10
    max_rt_tolerance: float = 0.15
    initial_rt_tolerance: float = 0.25
    fallback_rt_tolerance: float = 0.35
    max_fragments: int = 6
    max_precursor_isotopes: int = 1
    max_fragment_isotopes: int = 1


@dataclass
class RTBootstrapResult:
    calibrator: object  # exposes .predict(ref_rt) -> df[predicted_rt, rt_lb, rt_ub]
    success: bool
    fallback_reason: Optional[str]
    n_rounds: int = 0  # number of DIA windows searched
    n_evaluated: int = 0  # cumulative precursors matched across those windows
    n_unique_anchors: int = 0
    diagnostics: dict = field(default_factory=dict)


class AnchorSelector:
    """Deduplicated, q-value-filtered RT anchors from cumulative quick matches."""

    def __init__(self, cfg: RTBootstrapConfig):
        self.cfg = cfg

    def select(self, cumulative_df: pl.DataFrame) -> pl.DataFrame:
        """One anchor per peptidoform among targets passing the q-value cutoff.

        q-values are computed over all accumulated target and decoy matches;
        anchors need finite RTs and keep the highest score (ties: lowest
        precursor_index). No top-N fallback: too few anchors means keep
        collecting.
        """
        if cumulative_df.height == 0:
            return cumulative_df

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
            return target_df

        return target_df.sort(
            ["score", "precursor_index"], descending=[True, False]
        ).unique("peptidoform_index", keep="first")

    def is_sufficient(
        self, n_unique_anchors: int, min_unique_anchors: Optional[int] = None
    ) -> bool:
        threshold = (
            self.cfg.min_unique_anchors
            if min_unique_anchors is None
            else min_unique_anchors
        )
        return n_unique_anchors >= threshold

    def subsample_for_fit(self, anchors_df: pl.DataFrame) -> pl.DataFrame:
        """Deterministic random subsample, capped at `cfg.max_fit_anchors`."""
        if anchors_df.height <= self.cfg.max_fit_anchors:
            return anchors_df
        rng = np.random.RandomState(self.cfg.seed)
        keep_idx = np.sort(
            rng.choice(anchors_df.height, size=self.cfg.max_fit_anchors, replace=False)
        )
        return anchors_df[keep_idx.tolist()]


class DIARTBootstrapCalibrator:
    """Runs the initial DIA RT bootstrap described in the module docstring.

    `spec_reader` and `match_fn` are injection points for tests; leave them None.
    """

    def __init__(
        self,
        run,
        db_dir,
        ms2_tol_in_ppm: float = 10.0,
        cfg: Optional[RTBootstrapConfig] = None,
        figure_path: Optional[Path] = None,
        spec_reader: Optional[SpectralLibReader] = None,
        match_fn: Optional[Callable[[object, object], pl.DataFrame]] = None,
    ):
        self._dia_run = run
        self.ms2_tol_in_ppm = ms2_tol_in_ppm
        self.cfg = cfg or RTBootstrapConfig()
        self.figure_path = figure_path
        self._match_fn = match_fn or self._match_window
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
            min_rt_tolerance=self.cfg.min_rt_tolerance,
            max_rt_tolerance=self.cfg.max_rt_tolerance,
        )
        self.anchor_selector = AnchorSelector(self.cfg)
        initial_calibrator = LinearProjectionCalibrator(
            min_rt_in_seconds=self.min_rt_in_seconds,
            max_rt_in_seconds=self.max_rt_in_seconds,
            rt_tolerance=self.cfg.initial_rt_tolerance,
        ).fit(self.spec_reader.modification_df["ref_rt"])
        self.spec_reader.calibrate_rt(initial_calibrator)

    # one-time setup

    def _broad_fallback_calibrator(self) -> LinearProjectionCalibrator:
        calibrator = LinearProjectionCalibrator(
            min_rt_in_seconds=self.min_rt_in_seconds,
            max_rt_in_seconds=self.max_rt_in_seconds,
            rt_tolerance=self.cfg.fallback_rt_tolerance,
        ).fit(self.spec_reader.modification_df["ref_rt"])
        return calibrator

    def _make_bin_edges(self, ref_rt_eligible: np.ndarray) -> np.ndarray:
        """Bin edges for the diagnostic figure's coverage histogram only."""
        lo, hi = float(np.min(ref_rt_eligible)), float(np.max(ref_rt_eligible))
        return np.linspace(lo, hi if hi > lo else lo + 1.0, _PLOT_N_BINS + 1)

    # per-window matching

    def _match_window(self, dia_win, container) -> pl.DataFrame:
        """Quick-match every precursor of this DIA window (no sampling)."""
        num_precursors = container.precursor_mz_arr.shape[0]
        if num_precursors == 0:
            return pl.DataFrame(schema=_MATCH_SCHEMA)

        all_index0_arr = np.arange(num_precursors, dtype=np.int64)
        frame_index_arr, score_arr, valid_arr = _quick_match(
            container,
            dia_win.get_peak_container(),
            dia_win.get_frame_num_map(),
            all_index0_arr,
            fragment_mz_tol=self.ms2_tol_in_ppm,
        )
        if not np.any(valid_arr):
            return pl.DataFrame(schema=_MATCH_SCHEMA)

        peptidoform_index_arr = container.precursor_peptidoform_index_arr[valid_arr]
        peptide_index_arr = container.mod_peptide_index_arr[peptidoform_index_arr]
        observed_rt_arr = dia_win.meta_df[frame_index_arr[valid_arr]][
            "time_in_seconds"
        ].to_numpy()

        return pl.DataFrame(
            {
                "precursor_index": (
                    container.min_precursor_index + np.flatnonzero(valid_arr)
                ).astype(np.uint32),
                "peptidoform_index": peptidoform_index_arr.astype(np.uint32),
                "ref_rt": container.mod_ref_rt_arr[peptidoform_index_arr].astype(
                    np.float32
                ),
                "observed_rt": observed_rt_arr.astype(np.float32),
                "score": score_arr[valid_arr].astype(np.float32),
                "is_decoy": container.peptide_is_decoy_arr[peptide_index_arr].astype(
                    np.bool_
                ),
            },
            schema=_MATCH_SCHEMA,
        )

    # trial fitting

    def _try_fit(self, anchors_df: pl.DataFrame):
        fit_df = self.anchor_selector.subsample_for_fit(anchors_df)
        calibrator, diag = BootstrapRTCalibrator(**self._calibrator_kwargs).fit(
            fit_df["ref_rt"].to_numpy(),
            fit_df["observed_rt"].to_numpy(),
        )
        return calibrator, diag, fit_df

    def _top_score_fit(
        self, cumulative_df: pl.DataFrame
    ) -> Tuple[Optional[BootstrapRTCalibrator], dict, pl.DataFrame]:
        """Fit on the ``top_n_best_effort`` best-scoring targets, ignoring q-values.

        One match per peptidoform; `BootstrapRTCalibrator`'s QC still applies.
        Not called by `run()`.
        """
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

            ref_rt = self.spec_reader.modification_df["ref_rt"].to_numpy()
            ref_rt = ref_rt[np.isfinite(ref_rt)]
            bin_edges = (
                self._make_bin_edges(ref_rt)
                if ref_rt.size
                else np.linspace(0.0, 1.0, _PLOT_N_BINS + 1)
            )

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
            logger.info(f"Saved DIA RT bootstrap diagnostics: " f"{self.figure_path}")
        except Exception:
            logger.exception(f"Failed to save DIA RT bootstrap diagnostics")

    # orchestration

    def run(self) -> RTBootstrapResult:
        broad_calibrator = self._broad_fallback_calibrator()
        empty_df = pl.DataFrame(schema=_MATCH_SCHEMA)

        num_wins = self._dia_run.dia_scheme_df.shape[0]
        if num_wins == 0:
            return self._fallback_result(
                broad_calibrator,
                "no_dia_windows",
                anchors_df=empty_df,
                fit_df=empty_df,
            )

        win_indices = np.random.RandomState(self.cfg.seed).permutation(num_wins)
        min_windows = max(int(self.cfg.min_window_frac * num_wins), 3)

        cumulative_rounds: list[pl.DataFrame] = []
        n_windows, n_evaluated = 0, 0
        last_diag = {"reason": "no_windows_processed"}
        cumulative_df = empty_df
        anchors_df = empty_df
        fit_df = empty_df

        for win_idx in win_indices:
            dia_win = self._dia_run.get_dia_window(int(win_idx))
            if dia_win is None:
                continue
            container = self.spec_reader.read_by_mz_range(*dia_win.isolation_mz_range)
            if container is None:
                continue

            match_df = self._match_fn(dia_win, container)
            n_windows += 1
            n_evaluated += int(container.precursor_mz_arr.shape[0])
            if match_df.height > 0:
                cumulative_rounds.append(match_df)

            if cumulative_rounds:
                cumulative_df = pl.concat(cumulative_rounds, how="vertical")
                anchors_df = self.anchor_selector.select(cumulative_df)
                if self.anchor_selector.is_sufficient(anchors_df.height):
                    calibrator, diag, fit_df = self._try_fit(anchors_df)
                    last_diag = diag
                    if calibrator is not None:
                        logger.info(
                            f"DIA RT bootstrap succeeded: "
                            f"windows={n_windows} evaluated={n_evaluated} "
                            f"anchors={anchors_df.height} degree={diag.get('degree')} "
                            f"half_width={diag.get('half_width', float('nan')):.1f}s"
                        )
                        result = RTBootstrapResult(
                            calibrator=calibrator,
                            success=True,
                            fallback_reason=None,
                            n_rounds=n_windows,
                            n_evaluated=n_evaluated,
                            n_unique_anchors=anchors_df.height,
                            diagnostics=diag,
                        )
                        self._save_diagnostic_figure(
                            calibrator, anchors_df, fit_df, diag, success=True
                        )
                        return result

            # hard_limit applies only after min_windows windows
            if n_evaluated >= self.cfg.hard_limit and n_windows >= min_windows:
                break

        n_unique_anchors = anchors_df.height

        # budget or windows exhausted: one fit with the relaxed anchor floor
        # before the broad fallback
        if n_unique_anchors >= self.cfg.min_anchors_best_effort:
            calibrator, diag, fit_df = self._try_fit(anchors_df)
            last_diag = diag
            if calibrator is not None:
                logger.warning(
                    f"DIA RT bootstrap best-effort fit used "
                    f"(anchors={n_unique_anchors} below target={self.cfg.min_unique_anchors}): "
                    f"windows={n_windows} evaluated={n_evaluated} "
                    f"degree={diag.get('degree')} half_width={diag.get('half_width', float('nan')):.1f}s"
                )
                result = RTBootstrapResult(
                    calibrator=calibrator,
                    success=True,
                    fallback_reason=None,
                    n_rounds=n_windows,
                    n_evaluated=n_evaluated,
                    n_unique_anchors=n_unique_anchors,
                    diagnostics=diag,
                )
                self._save_diagnostic_figure(
                    calibrator, anchors_df, fit_df, diag, success=True
                )
                return result

        if n_windows == 0:
            reason = "no_windows_processed"
            last_diag = {"reason": reason}
        elif not cumulative_rounds:
            reason = "no_matches_found"
            last_diag = {"reason": reason}
        elif n_unique_anchors < self.cfg.min_anchors_best_effort:
            reason = "insufficient_qualified_anchors"
            last_diag = {"reason": reason}
        else:
            reason = last_diag.get(
                "reason", "budget_or_windows_exhausted_without_valid_fit"
            )
        return self._fallback_result(
            broad_calibrator,
            reason,
            n_windows,
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
            f"DIA RT bootstrap failed, using broad fallback bounds: "
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
