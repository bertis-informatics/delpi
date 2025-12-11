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
from delpi.utils.numeric import cosine_similarity_columns
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
    fragment_mz_tol: float = 10.0,
    # similarity_cutoff=0.8,
):

    num_fragments = speclib_container.max_fragments
    num_precursors = speclib_container.precursor_mz_arr.shape[0]
    frame_index_arr = np.empty(num_precursors, dtype=np.int32)
    score_arr = np.empty(num_precursors, dtype=np.float32)

    for precursor_index0 in nb.prange(num_precursors):
        min_frame_index, max_frame_index = get_frame_index_range(
            speclib_container, frame_num_map.ms2_rt_arr, precursor_index0
        )

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

        similarity_scores = cosine_similarity_columns(xic_arr, fragment_intensity_arr)
        j = np.argmax(similarity_scores)

        frame_index_arr[precursor_index0] = min_frame_index + j
        score_arr[precursor_index0] = similarity_scores[j] + 0.1 * np.count_nonzero(
            xic_arr[:, j - 1 : j + 2]
        )

    return frame_index_arr, score_arr


def run_quick_search(
    run: DIARun,
    db_dir: Path,
    ms2_tol_in_ppm: float = 10,
    min_matches: int = 500000,
    q_value_cutoff: float = 0.05,
):

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

        frame_num_map = dia_win.get_frame_num_map()
        ms2_peak_df = dia_win.get_peak_container()

        frame_index_arr, score_arr = _quick_match(
            speclib_container,
            ms2_peak_df,
            frame_num_map,
            fragment_mz_tol=ms2_tol_in_ppm,
        )
        # score_cutoff = np.median(score_arr, axis=0)
        # mask = np.all(score_arr > score_cutoff, axis=1)
        # mask = score_arr > np.median(score_arr)
        mask = score_arr > 1.0
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
