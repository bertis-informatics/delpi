from collections import defaultdict
from pathlib import Path

import numpy as np
import numba as nb
import polars as pl

from delpi.lcms.dia_run import DIARun
from delpi.database.spec_lib_reader import SpectralLibReader
from delpi.database.numba.spec_lib_container import SpectralLibContainer
from delpi.lcms.data_container import DIAWindowFrameNumMap, PeakContainer
from delpi.model.rt_calibrator import LinearProjectionCalibrator
from delpi.search.dia.peak_group import make_xic_array
from delpi.utils.fdr import calculate_q_value
from delpi.utils.numeric import (
    cosine_similarity_columns,
    corrcoef,
    extract_upper_triangle,
)
from delpi.utils.peak import find_peak_index
from delpi.database.numba.spec_lib_utils import (
    get_frame_index_range,
    get_theoretical_peaks,
)


@nb.njit(parallel=True, fastmath=True, cache=True)
def _quick_match(
    speclib_container: SpectralLibContainer,
    ms2_peak_df: PeakContainer,
    frame_num_map: DIAWindowFrameNumMap,
    precursor_index0_arr: np.ndarray,
    fragment_mz_tol: float = 10.0,
    # similarity_cutoff=0.8,
):
    """Quick-match a (possibly strict subset of) local precursor indices.

    ``precursor_index0_arr`` selects which local precursors (0-based, into
    ``speclib_container``) to evaluate; outputs are aligned 1:1 with it
    (not with the full container). ``valid_arr[i]`` is False when precursor
    ``precursor_index0_arr[i]`` has no usable RT/XIC window (empty or
    inverted RT range, or an out-of-bounds frame index) -- callers must
    filter by ``valid_arr`` before using ``frame_index_arr``/``score_arr``.
    """

    num_fragments = speclib_container.max_fragments
    n_ms2_frames = frame_num_map.ms2_rt_arr.shape[0]
    n_sel = precursor_index0_arr.shape[0]
    frame_index_arr = np.full(n_sel, -1, dtype=np.int32)
    score_arr = np.full(n_sel, -np.inf, dtype=np.float32)
    valid_arr = np.zeros(n_sel, dtype=np.bool_)

    for i in nb.prange(n_sel):
        precursor_index0 = precursor_index0_arr[i]
        min_frame_index, max_frame_index = get_frame_index_range(
            speclib_container, frame_num_map.ms2_rt_arr, precursor_index0
        )
        # empty/inverted RT window (e.g. predicted RT bound outside the run)
        if min_frame_index >= n_ms2_frames or max_frame_index < min_frame_index:
            continue

        theo_peaks = get_theoretical_peaks(speclib_container, precursor_index0)
        fragment_mz_arr = theo_peaks.fragment_mz_arr
        fragment_intensity_arr = theo_peaks.fragment_intensity_arr

        ms2_st, ms2_ed = find_peak_index(
            ms2_peak_df.mz_arr, fragment_mz_arr.flatten(), fragment_mz_tol
        )
        # [#XICs, #RT-window]
        xic_arr = make_xic_array(
            ms2_peak_df,
            frame_num_map,
            ms2_st,
            ms2_ed,
            min_frame_index,
            max_frame_index,
            num_fragments,
        )
        if xic_arr.shape[1] == 0:
            continue

        similarity_scores = cosine_similarity_columns(xic_arr, fragment_intensity_arr)
        j = np.argmax(similarity_scores)
        # the 5-frame co-elution window (center +/- 2) needed for the
        # fragment-correlation term must fit inside the XIC array
        if j < 2 or j >= xic_arr.shape[1] - 2:
            continue

        # peak count over the center +/- 1 neighborhood, normalized to [0, 1]
        lo, hi = j - 1, j + 2
        peak_count_score = np.count_nonzero(xic_arr[:, lo:hi]) / (num_fragments * 3)

        # mean cubed pairwise correlation between co-eluting fragment XICs
        # (center +/- 2), restricted to fragments with a non-zero peak in
        # that window
        xic_window = xic_arr[:, j - 2 : j + 3]
        qualified_mask = np.zeros(num_fragments, dtype=np.bool_)
        n_qualified = 0
        for r in range(num_fragments):
            if np.any(xic_window[r] != 0):
                qualified_mask[r] = True
                n_qualified += 1

        xic_corr_score = 0.0
        if n_qualified >= 2:
            qualified_xic = np.empty(
                (n_qualified, xic_window.shape[1]), dtype=xic_window.dtype
            )
            idx = 0
            for r in range(num_fragments):
                if qualified_mask[r]:
                    qualified_xic[idx] = xic_window[r]
                    idx += 1
            corr_mat = corrcoef(qualified_xic)
            xic_corr_score = np.mean(extract_upper_triangle(corr_mat) ** 3)

        frame_index_arr[i] = min_frame_index + j
        score_arr[i] = similarity_scores[j] + xic_corr_score + peak_count_score
        valid_arr[i] = True

    return frame_index_arr, score_arr, valid_arr


def run_quick_search(
    run: DIARun,
    db_dir: Path,
    ms2_tol_in_ppm: float = 10,
    min_matches: int = 500000,
    q_value_cutoff: float = 0.05,
):
    """Legacy match-count-based quick search (kept for backward compatibility
    and as the fallback path for engines that don't implement
    ``perform_rt_bootstrap``). The primary initial-DIA-calibration path now
    uses the RT-stratified, checkpointed bootstrap in
    :mod:`delpi.search.dia.rt_bootstrap`, which calls :func:`_quick_match`
    per round on sampled precursor subsets.
    """

    num_wins = run.dia_scheme_df.shape[0]
    win_indices = np.random.RandomState(seed=1226).permutation(num_wins)

    ############# first (coarse) pass for RT mapping ####################
    spec_reader = SpectralLibReader(
        peptide_db_path=db_dir,
        max_fragments=6,
        max_precursor_isotopes=1,
        max_fragment_isotopes=1,
    )

    ## Set initial retention time tolerance
    rt_calibrator = LinearProjectionCalibrator(
        min_rt_in_seconds=0,
        max_rt_in_seconds=run.gradient_length_in_seconds,
        rt_tolerance=0.3333,
    ).fit(spec_reader.modification_df["ref_rt"])
    spec_reader.calibrate_rt(rt_calibrator)

    ## Fitting reference RT to observed RT
    match_results = defaultdict(list)
    num_matches = 0
    for win_idx in win_indices:
        dia_win = run.get_dia_window(win_idx)
        speclib_container = spec_reader.read_by_mz_range(*dia_win.isolation_mz_range)

        if speclib_container is None:
            continue

        frame_num_map = dia_win.get_frame_num_map()
        ms2_peak_df = dia_win.get_peak_container()

        num_precursors = speclib_container.precursor_mz_arr.shape[0]
        all_index0_arr = np.arange(num_precursors, dtype=np.int64)
        frame_index_arr, score_arr, valid_arr = _quick_match(
            speclib_container,
            ms2_peak_df,
            frame_num_map,
            all_index0_arr,
            fragment_mz_tol=ms2_tol_in_ppm,
        )
        # score_cutoff = np.median(score_arr, axis=0)
        # mask = np.all(score_arr > score_cutoff, axis=1)
        # mask = score_arr > np.median(score_arr)
        mask = valid_arr & (score_arr > 1.0)
        precursor_index0_arr = np.flatnonzero(mask).astype(np.uint32)
        frame_index_arr = frame_index_arr[mask]
        score_arr = score_arr[mask]
        peptidoform_index_arr = speclib_container.precursor_peptidoform_index_arr[
            precursor_index0_arr
        ]

        match_results["peptidoform_index"].append(peptidoform_index_arr)
        match_results["score"].append(score_arr)
        match_results["observed_rt"].append(
            dia_win.meta_df[frame_index_arr]["time_in_seconds"].to_numpy()
        )

        num_matches += precursor_index0_arr.shape[0]
        if num_matches > min_matches:
            break

    for k, v in match_results.items():
        match_results[k] = np.concatenate(v)

    df = pl.DataFrame(match_results)
    peptide_index_arr = speclib_container.mod_peptide_index_arr[df["peptidoform_index"]]
    is_decoy_arr = speclib_container.peptide_is_decoy_arr[peptide_index_arr]
    df = df.with_columns(is_decoy=is_decoy_arr)

    df = calculate_q_value(df, score_column="score", out_column="precursor_q_value")
    t_df = df.filter(
        (pl.col("is_decoy") == False) & (pl.col("precursor_q_value") < q_value_cutoff)
    )

    if t_df.shape[0] < 1000:
        t_df = (
            df.filter(pl.col("is_decoy") == False)
            .sort(pl.col("score"), descending=True)
            .limit(1000)
        )

    ref_rt_arr = speclib_container.mod_ref_rt_arr[t_df["peptidoform_index"]]
    t_df = t_df.with_columns(ref_rt=ref_rt_arr)

    return t_df
