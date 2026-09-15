"""MS2-only PmSM proposals and inexpensive features for DIA RT bootstrap.

Candidate selection follows ``find_peak_groups``: fragment-XIC maxima, weighted
clustering, a matched-cell threshold, and top-k co-elution support. Only
retained candidates receive the more expensive correlation/ppm features.

No final identification score, target/decoy filtering, or RT fitting is
performed here. Score candidates downstream, then select one winner per
precursor before estimating precursor-level q-values.
"""

from typing import NamedTuple

import numba as nb
import numpy as np

from delpi.constants import QUANT_FRAGMENTS
from delpi.database.numba.spec_lib_container import SpectralLibContainer
from delpi.lcms.data_container import DIAWindowFrameNumMap, PeakContainer
from delpi.search.dia.peak_group import PeakGroupContainer, count_xic_peaks
from delpi.utils.numeric import corrcoef, extract_upper_triangle
from delpi.utils.peak import find_peak_index
from delpi.utils.signal import cluster_peaks_with_weights

QUICK_PMSM_FEATURE_NAMES = (
    "nonzero_peak_count",
    "center_cosine_similarity",
    "xic_corr_score",
    "mean_abs_mass_error_ppm",
)
_N_FEATURES = len(QUICK_PMSM_FEATURE_NAMES)


class QuickPmSMMatchContainer(NamedTuple):
    """Row-aligned candidate identities and float32 scoring features.

    ``peak_groups.precursor_index0_arr`` contains indices into the supplied
    library container, not peptidoform IDs. Add ``min_precursor_index`` to
    obtain global precursor IDs under the existing contiguous-slice library
    contract. ``frame_num_arr`` contains actual center MS2 frame numbers,
    NOT local frame-array indices.

    ``peak_groups.peak_count_arr`` is deliberately an empty uint16 array:
    use ``features[:, 0]`` for MS2 matched-cell counts. Do not pass this
    identity-only container to code requiring a populated peak-count array.

    ``features`` has shape (number_of_candidates, 4), including (0, 4)
    for an empty result. Columns are ``QUICK_PMSM_FEATURE_NAMES``.
    """

    peak_groups: PeakGroupContainer
    features: np.ndarray

    @property
    def feature_names(self) -> tuple[str, ...]:
        return QUICK_PMSM_FEATURE_NAMES


@nb.njit(nogil=True, cache=True)
def _make_xic_with_peak_indices(
    ms2_peak_df,
    frame_num_map,
    fragment_mz,
    min_frame_index,
    max_frame_index,
    ms2_mass_tol,
):
    """Build each mono-fragment XIC once; remember its winning centroid.

    Highest positive finite intensity wins within a fragment/frame cell,
    as in make_xic_array. Equal intensities retain the first centroid in
    the input m/z-sorted array (not the centroid closest in mass).
    """
    n_fragments = fragment_mz.size
    n_frames = max_frame_index - min_frame_index + 1
    xic = np.zeros((n_fragments, n_frames), dtype=np.float32)
    peak_indices = np.zeros((n_fragments, n_frames), dtype=np.uint32)
    # Invalid theoretical masses must never enter searchsorted. Zero is
    # only a placeholder for such rows; the matching loop skips them.
    query_mz = fragment_mz.copy()
    for f in range(n_fragments):
        if not np.isfinite(query_mz[f]) or query_mz[f] <= 0:
            query_mz[f] = 0.0
    starts, stops = find_peak_index(ms2_peak_df.mz_arr, query_mz, ms2_mass_tol)
    n_map = frame_num_map.frame_num_to_index_arr.size
    for f in range(n_fragments):
        if query_mz[f] <= 0:
            continue
        for p in range(starts[f], stops[f]):
            frame_num = np.int64(ms2_peak_df.frame_num_arr[p])
            if frame_num < 0 or frame_num >= n_map:
                continue
            fi = np.int64(frame_num_map.frame_num_to_index_arr[frame_num])
            if fi < min_frame_index or fi > max_frame_index:
                continue
            # A shared frame map can also contain MS1 entries. Require an
            # actual frame from this MS2 window, not only a valid index.
            if frame_num_map.ms2_frame_num_arr[fi] != frame_num:
                continue
            abundance = ms2_peak_df.ab_arr[p]
            if not np.isfinite(abundance) or abundance <= 0:
                continue
            # XIC intensity follows the project's float32 convention.
            abundance32 = np.float32(abundance)
            if not np.isfinite(abundance32):
                continue
            t = fi - min_frame_index
            if abundance32 > xic[f, t]:
                xic[f, t] = abundance32
                peak_indices[f, t] = p
    return xic, peak_indices


@nb.njit(nogil=True, cache=True)
def _extract_features(
    xic, peak_indices, observed_mz, fragment_mz, predicted, center, radius
):
    """Compute four features for one already selected candidate."""
    n_fragments = xic.shape[0]
    lo, hi = center - radius, center + radius + 1
    width = hi - lo
    features = np.zeros(_N_FEATURES, dtype=np.float32)
    n_nonzero = 0
    n_qualified = 0
    dot = 0.0
    obs_norm2 = 0.0
    pred_norm2 = 0.0
    abs_ppm_sum = 0.0

    # Float64 accumulation avoids cancellation in ppm variance and keeps
    # cosine stable even for large, raw intensity values.
    for f in range(n_fragments):
        obs = np.float64(xic[f, center])
        pred = np.float64(predicted[f])
        if not np.isfinite(pred) or pred < 0:
            pred = 0.0
        dot += obs * pred
        obs_norm2 += obs * obs
        pred_norm2 += pred * pred
        present = False
        for t in range(lo, hi):
            if xic[f, t] <= 0:
                continue
            present = True
            p = peak_indices[f, t]
            mz = np.float64(fragment_mz[f])
            ppm = (np.float64(observed_mz[p]) - mz) * 1e6 / mz
            n_nonzero += 1
            abs_ppm_sum += abs(ppm)
        if present:
            n_qualified += 1

    features[0] = n_nonzero
    if obs_norm2 > 0 and pred_norm2 > 0:
        features[1] = min(1.0, max(0.0, dot / np.sqrt(obs_norm2 * pred_norm2)))
    if n_qualified >= 2:
        # Same definition as _quick_match: mean(r_ij ** 3), excluding
        # absent rows and the diagonal. Constant nonzero rows stay in the
        # denominator and have zero off-diagonal correlation.
        qualified = np.empty((n_qualified, width), dtype=np.float64)
        out_row = 0
        for f in range(n_fragments):
            present = False
            for t in range(lo, hi):
                if xic[f, t] > 0:
                    present = True
                    break
            if present:
                for k in range(width):
                    qualified[out_row, k] = xic[f, lo + k]
                out_row += 1
        correlations = extract_upper_triangle(corrcoef(qualified))
        total = 0.0
        for r in correlations:
            # Bound only roundoff excursions; preserve negative evidence.
            r = min(1.0, max(-1.0, r))
            total += r * r * r
        features[2] = total / correlations.size
    if n_nonzero > 0:
        features[3] = abs_ppm_sum / n_nonzero
    return features


@nb.njit(parallel=True, nogil=True, cache=True)
def find_quick_pmsm_matches(
    speclib_container: SpectralLibContainer,
    ms2_peak_df: PeakContainer,
    frame_num_map: DIAWindowFrameNumMap,
    ms2_mass_tol: float = 10,
    rt_window_radius: int = 2,
    min_xic_peak_count: int = 3,
    min_peak_count: int = 6,
    topk: int = 3,
) -> QuickPmSMMatchContainer:
    """Return up to ``topk`` MS2-only PmSM candidates per input precursor.

    The default PmSM has exactly five consecutive spectra of this DIA
    window (center +/- 2). Truncated neighborhoods are not returned.
    Only monoisotopic fragment peaks are used, irrespective of the library's
    precursor/fragment isotope settings. All supplied fragment rows are
    considered; no additional predicted-intensity threshold is imposed.

    Candidate selection mirrors find_peak_groups' MS2-fragment path:
    XIC apex support -> weighted clustering (distance <= 2 frames) ->
    min_peak_count -> topk by the same three-frame co-elution proxy.
    Ties prefer the earlier frame. Correlation and mass features are
    calculated only for retained topk candidates, not for every RT frame.

    ``min_peak_count`` counts occupied fragment/frame cells across the
    PmSM, not all centroids falling inside a mass tolerance. Several
    centroids matching the same cell contribute once. A centroid matching
    multiple theoretical fragments can still occupy multiple cells, as
    in the existing fragment-XIC construction; this is not a count of
    globally unique raw peaks.

    Features, in order:
      0. Nonzero fragment/frame count over the complete PmSM.
      1. Center-spectrum cosine with the theoretical fragment intensities.
      2. Mean cubed pairwise fragment-XIC Pearson correlation over the
         complete PmSM, on rows containing at least one nonzero value.
      3. Unweighted mean absolute ppm error of occupied cells.
      4. Number of observed fragments in the center spectrum.
      5. Population standard deviation of signed ppm errors (ddof=0).

    Preconditions: centroid arrays are aligned and m/z-sorted; MS2 RTs
    are finite, sorted, and aligned with the frame map; the library uses
    the existing contiguous local precursor/fragment layout. Large sorted
    input arrays are not rescanned for validation on each invocation.
    Missing RT bounds (NaN) leave that side unbounded. Inverted/outside-run
    ranges, short XICs, and precursors with no candidate return no rows.
    """
    if not np.isfinite(ms2_mass_tol) or ms2_mass_tol <= 0:
        raise ValueError("ms2_mass_tol must be finite and positive")
    if rt_window_radius < 1 or rt_window_radius != int(rt_window_radius):
        raise ValueError("rt_window_radius must be an integer >= 1")
    if min_xic_peak_count < 1 or min_xic_peak_count != int(min_xic_peak_count):
        raise ValueError("min_xic_peak_count must be an integer >= 1")
    if min_peak_count < 1 or min_peak_count != int(min_peak_count):
        raise ValueError("min_peak_count must be an integer >= 1")
    if topk < 0 or topk != int(topk):
        raise ValueError("topk must be a nonnegative integer")
    radius = int(rt_window_radius)
    k_max = int(topk)
    n_precursors = speclib_container.precursor_mz_arr.size
    n_fragments = int(speclib_container.max_fragments)
    n_ms2_frames = frame_num_map.ms2_rt_arr.size
    if n_fragments < 0:
        raise ValueError("max_fragments must be nonnegative")
    if frame_num_map.ms2_frame_num_arr.size != n_ms2_frames:
        raise ValueError("MS2 frame numbers and RT arrays must be aligned")
    n_peaks = ms2_peak_df.mz_arr.size
    if ms2_peak_df.frame_num_arr.size != n_peaks or ms2_peak_df.ab_arr.size != n_peaks:
        raise ValueError("MS2 m/z, frame-number and intensity arrays must be aligned")
    if n_peaks > 4294967295:
        raise ValueError("find_peak_index requires uint32-compatible peak indices")
    if speclib_container.precursor_peptidoform_index_arr.size != n_precursors:
        raise ValueError("Precursor metadata arrays must be aligned")
    required = n_precursors * n_fragments
    if (
        speclib_container.speclib_mz_arr.size != required
        or speclib_container.speclib_predicted_intensity_arr.size != required
    ):
        raise ValueError("Expected max_fragments flattened rows per local precursor")

    if (
        n_precursors == 0
        or n_fragments == 0
        or k_max == 0
        or n_peaks == 0
        or n_ms2_frames < 2 * radius + 1
        or min_xic_peak_count > n_fragments
    ):
        return QuickPmSMMatchContainer(
            PeakGroupContainer(
                speclib_container.min_precursor_index,
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint16),
            ),
            np.empty((0, _N_FEATURES), dtype=np.float32),
        )

    # Each worker owns one precursor row. Scratch XICs are local to that
    # worker; do not allocate a precursor x fragment x run tensor.
    group_counts = np.zeros(n_precursors, dtype=np.int64)
    frame_buffer = np.empty((n_precursors, k_max), dtype=np.uint32)
    feature_buffer = np.empty((n_precursors, k_max, _N_FEATURES), dtype=np.float32)
    for precursor_index0 in nb.prange(n_precursors):
        mod_idx = np.int64(
            speclib_container.precursor_peptidoform_index_arr[precursor_index0]
        )
        if (
            mod_idx < 0
            or mod_idx >= speclib_container.mod_rt_lb_arr.size
            or mod_idx >= speclib_container.mod_rt_ub_arr.size
        ):
            continue
        rt_lb = speclib_container.mod_rt_lb_arr[mod_idx]
        rt_ub = speclib_container.mod_rt_ub_arr[mod_idx]
        if rt_lb > rt_ub:
            continue
        min_frame = (
            0
            if np.isnan(rt_lb)
            else np.int64(np.searchsorted(frame_num_map.ms2_rt_arr, rt_lb, side="left"))
        )
        # Convert the exclusive searchsorted stop to an INCLUSIVE index.
        max_frame = (
            n_ms2_frames - 1
            if np.isnan(rt_ub)
            else np.int64(
                np.searchsorted(frame_num_map.ms2_rt_arr, rt_ub, side="right")
            )
            - 1
        )
        xic_len = max_frame - min_frame + 1
        if xic_len < 2 * radius + 1:
            continue

        # Same monoisotopic data as get_theoretical_peaks(...).fragment_mz_arr[0],
        # without constructing unused precursor/isotope theoretical arrays.
        st = precursor_index0 * n_fragments
        fragment_mz = speclib_container.speclib_mz_arr[st : st + n_fragments]
        predicted = speclib_container.speclib_predicted_intensity_arr[
            st : st + n_fragments
        ]
        xic, peak_indices = _make_xic_with_peak_indices(
            ms2_peak_df, frame_num_map, fragment_mz, min_frame, max_frame, ms2_mass_tol
        )
        xic_counts = count_xic_peaks(xic)
        spec_counts = np.zeros(xic_len, dtype=np.int64)
        quant_counts = np.zeros(xic_len, dtype=np.int64)
        quant_start = max(0, n_fragments - QUANT_FRAGMENTS)
        for f in range(n_fragments):
            for t in range(xic_len):
                if xic[f, t] > 0:
                    spec_counts[t] += 1
                    if f >= quant_start:
                        quant_counts[t] += 1
        combined = spec_counts + xic_counts
        candidate_centers = np.flatnonzero(
            (xic_counts >= min_xic_peak_count) & (quant_counts > 0)
        ).astype(np.int64)
        # Apply full-neighborhood bounds BEFORE clustering so an invalid
        # edge winner cannot suppress an otherwise valid nearby candidate.
        candidate_centers = candidate_centers[
            (candidate_centers >= radius) & (candidate_centers < xic_len - radius)
        ]
        if candidate_centers.size == 0:
            continue
        weights = (
            combined[candidate_centers - 1]
            + 2 * combined[candidate_centers]
            + combined[candidate_centers + 1]
        )
        candidate_centers = cluster_peaks_with_weights(
            candidate_centers,
            weights,
            dist_cutoff=2,
            min_cluster_size=1,
            max_cluster_size=1024,
        )
        # Prefix sums give an O(1) full-PmSM count for each candidate.
        prefix_counts = np.empty(xic_len + 1, dtype=np.int64)
        prefix_counts[0] = 0
        prefix_counts[1:] = np.cumsum(spec_counts)
        counts = (
            prefix_counts[candidate_centers + radius + 1]
            - prefix_counts[candidate_centers - radius]
        )
        candidate_centers = candidate_centers[counts >= min_peak_count]
        if candidate_centers.size == 0:
            continue
        weights = (
            combined[candidate_centers - 1]
            + 2 * combined[candidate_centers]
            + combined[candidate_centers + 1]
        )
        # Sorted centers + stable descending weights -> deterministic ties.
        order = np.argsort(-weights, kind="mergesort")
        n_keep = min(k_max, candidate_centers.size)
        for k in range(n_keep):
            center = candidate_centers[order[k]]
            frame_buffer[precursor_index0, k] = frame_num_map.ms2_frame_num_arr[
                min_frame + center
            ]
            feature_buffer[precursor_index0, k] = _extract_features(
                xic,
                peak_indices,
                ms2_peak_df.mz_arr,
                fragment_mz,
                predicted,
                center,
                radius,
            )
        group_counts[precursor_index0] = n_keep

    offsets = np.empty(n_precursors + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(group_counts)
    n_matches = offsets[-1]
    precursor_indices = np.empty(n_matches, dtype=np.uint32)
    frame_numbers = np.empty(n_matches, dtype=np.uint32)
    features = np.empty((n_matches, _N_FEATURES), dtype=np.float32)
    for i in nb.prange(n_precursors):
        for k in range(group_counts[i]):
            out_idx = offsets[i] + k
            precursor_indices[out_idx] = i
            frame_numbers[out_idx] = frame_buffer[i, k]
            features[out_idx] = feature_buffer[i, k]
    return QuickPmSMMatchContainer(
        PeakGroupContainer(
            speclib_container.min_precursor_index,
            precursor_indices,
            frame_numbers,
            np.empty(0, dtype=np.uint16),
        ),
        features,
    )
