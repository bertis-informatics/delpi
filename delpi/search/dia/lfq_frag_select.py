import numpy as np
import numba as nb

from delpi.utils.numeric import rowwise_pearsonr, pearsonr


@nb.njit(nogil=True, fastmath=True, cache=True)
def get_consensus_xic(xic_arr):
    """
    xic_arr: [n_frag, n_time]

    return: Median consensus XIC after 95th percentile normalization
    """
    n_frag, n_time = xic_arr.shape
    epsilon = 1e-9
    norm_buf = np.empty((n_frag, n_time), dtype=np.float32)
    valid_cnt = 0

    for i in range(n_frag):
        row = xic_arr[i]
        if np.sum(row) < epsilon:
            continue

        # 2. 95th Percentile
        sorted_row = np.sort(row)
        scale = sorted_row[int(0.95 * (n_time - 1))]

        if scale < epsilon:
            continue

        # normalization
        norm_buf[valid_cnt] = row / scale
        valid_cnt += 1

    if valid_cnt == 0:
        return np.zeros(n_time, dtype=np.float32)

    # 3. Column-wise median calculation
    consensus_xic = np.empty(n_time, dtype=np.float32)
    col_buf = np.empty(valid_cnt, dtype=np.float32)

    mid_idx = valid_cnt // 2
    is_odd = valid_cnt % 2 == 1

    for t in range(n_time):
        col_buf[:] = norm_buf[:valid_cnt, t]
        col_buf.sort()

        if is_odd:
            consensus_xic[t] = col_buf[mid_idx]
        else:
            consensus_xic[t] = 0.5 * (col_buf[mid_idx - 1] + col_buf[mid_idx])

    return consensus_xic


@nb.njit(nogil=True, fastmath=True, cache=True)
def get_representative_xic(xic_arr):
    """
    Select the fragment XIC with the highest average correlation to all other fragments.
    Similar to DIA-NN's approach for finding representative XICs.

    Args:
        xic_arr: [n_frag, n_time] - Array of fragment XICs

    Returns:
        numpy.array: Representative XIC [n_time] - the fragment with highest avg correlation
    """
    n_frag, n_time = xic_arr.shape
    epsilon = 1e-9

    # Filter out zero-intensity fragments
    valid_indices = np.empty(n_frag, dtype=np.int32)
    valid_cnt = 0

    for i in range(n_frag):
        if np.sum(xic_arr[i]) > epsilon:
            valid_indices[valid_cnt] = i
            valid_cnt += 1

    if valid_cnt == 0:
        return np.zeros(n_time, dtype=np.float32)

    if valid_cnt == 1:
        return xic_arr[valid_indices[0]].copy()

    # Calculate average correlation for each valid fragment
    avg_corr = np.zeros(valid_cnt, dtype=np.float32)

    for i in range(valid_cnt):
        frag_i = valid_indices[i]
        xic_i = xic_arr[frag_i]
        total_corr = 0.0

        for j in range(valid_cnt):
            if i == j:
                continue
            frag_j = valid_indices[j]
            xic_j = xic_arr[frag_j]

            # Compute Pearson correlation
            corr = pearsonr(xic_i, xic_j)
            total_corr += corr**3

        # Average correlation (excluding self)
        avg_corr[i] = total_corr / (valid_cnt - 1)

    # Find fragment with highest average correlation
    best_idx = 0
    best_corr = avg_corr[0]

    for i in range(1, valid_cnt):
        if avg_corr[i] > best_corr:
            best_corr = avg_corr[i]
            best_idx = i

    # Return the representative fragment XIC
    return xic_arr[valid_indices[best_idx]].copy()


@nb.njit(nogil=True, fastmath=True, cache=True)
def select_quantifiable_fragments_by_avg_corr(
    xic_arrays,
    target_fragments=9,
    max_fragments=12,
    corr_thresh=0.9,
    cube_corr=False,
    rep_type=0,
):
    """
    Select fragments based on average correlation with consensus XIC across runs.

    This approach computes the average Pearson correlation of each fragment
    with the consensus XIC across all runs, then selects fragments with correlation
    above threshold, bounded by min and max fragment counts.

    Args:
        xic_arrays (numpy.array): [n_runs, n_frags, n_time]
        target_fragments (int, optional): number of fragments this selection
            aims for -- NOT a hard cutoff for downstream quantity reporting
            (see `filter_fragments_by_cross_run_intensity`'s
            `min_quant_fragments` for that). Defaults to 9.
        max_fragments (int, optional): maximum number of fragments to select. Defaults to 12.
        corr_thresh (float, optional): minimum correlation threshold. Defaults to 0.9.
        cube_corr (bool, optional): whether to cube correlations. Defaults to False.
        rep_type (int, optional): 0 for consensus, 1 for representative. Defaults to 0.

    Returns:
        numpy.array: Indices of selected fragments sorted by average
        correlation (descending); an empty array if there are no valid
        (non-near-zero-intensity) fragments at all.
    """
    n_runs, n_frags, n_time = xic_arrays.shape
    epsilon = 1e-6

    # 1. Initial Filtering: Remove fragments with near-zero total intensity across all runs
    total_intensities = np.zeros(n_frags, dtype=np.float32)
    for frag_idx in range(n_frags):
        total = 0.0
        for run_idx in range(n_runs):
            for t in range(n_time):
                total += xic_arrays[run_idx, frag_idx, t]
        total_intensities[frag_idx] = total

    # Collect valid fragment indices
    valid_indices = np.empty(n_frags, dtype=np.int32)
    valid_len = 0
    for frag_idx in range(n_frags):
        if total_intensities[frag_idx] > epsilon:
            valid_indices[valid_len] = frag_idx
            valid_len += 1

    if valid_len == 0:
        return np.empty(0, dtype=np.int32)

    # 2. Calculate average correlation for each valid fragment across all runs
    avg_correlations = np.zeros(valid_len, dtype=np.float32)

    for run_idx in range(n_runs):
        # Get XICs for valid fragments in current run
        current_xics = xic_arrays[run_idx, valid_indices[:valid_len], :]

        # Build consensus XIC from current run
        if rep_type == 0:
            consensus = get_consensus_xic(current_xics)
        else:
            consensus = get_representative_xic(current_xics)

        # Compute correlations between each fragment and the consensus
        run_corrs = rowwise_pearsonr(current_xics, consensus)

        # Apply penalty to negative correlations and optional cubing
        for i in range(valid_len):
            corr = run_corrs[i]
            # Apply cubing if requested
            if cube_corr:
                if corr < 0:
                    # Preserve sign when cubing negative values
                    corr = -((-corr) ** 3)
                else:
                    corr = corr**3

            avg_correlations[i] += corr

    # Average across runs
    for i in range(valid_len):
        avg_correlations[i] = avg_correlations[i] / n_runs

    # 3. Sort fragments by average correlation (descending)
    sorted_order = np.argsort(-avg_correlations)  # negative for descending order

    # 4. Select fragments based on correlation threshold and bounds
    # First, count how many fragments meet the threshold
    above_threshold_count = 0
    for i in range(valid_len):
        if avg_correlations[sorted_order[i]] >= corr_thresh:
            above_threshold_count += 1
        else:
            break  # Since sorted, no more will meet threshold

    if above_threshold_count >= target_fragments:
        # We have enough fragments above threshold
        # Select up to max_fragments among those above threshold
        selected_count = min(above_threshold_count, max_fragments)
    else:
        # Not enough fragments above threshold -- aim for target_fragments,
        # bounded by both how many valid fragments exist and max_fragments
        selected_count = min(target_fragments, valid_len, max_fragments)

    # Build the selected indices array
    selected_indices = np.empty(selected_count, dtype=np.int32)
    for i in range(selected_count):
        selected_indices[i] = valid_indices[sorted_order[i]]

    return selected_indices


@nb.njit(cache=True, nogil=True)
def filter_fragments_by_cross_run_intensity(
    xic_arrays: np.ndarray,
    selected_indices: np.ndarray,
    min_quant_fragments: int,
    min_interference_runs: int,
    interference_min_log2_fold: float,
    interference_z_threshold: float,
    epsilon: float,
) -> np.ndarray:
    """Drop shape-correlated fragments whose relative intensity is an outlier in only some runs.

    Shape correlation (`select_quantifiable_fragments_by_avg_corr`) is scale
    invariant, so a fragment can pass it even if one run's intensity is
    contaminated (e.g. co-eluting interference, or partial signal loss) while
    its XIC shape still looks normal. This detects that case using only
    *relative*, run-centered fragment intensities (never a fragment's/run's
    absolute level), so it never flags a fragment for being globally weak,
    nor a run for having globally high/low precursor abundance (see the
    module's fixed integration window -- `xic_arrays`'s last axis -- carried
    over unchanged from the caller). Both unusually *high* and unusually
    *low* run-specific relative fragment intensities are treated as
    interference (two-sided); a fragment that is consistently high or low
    across every run is never flagged, since that's absorbed by the
    per-fragment cross-run reference (`B_k` below).

    ``xic_arrays``: this precursor's ``(n_runs, n_frags, n_time)`` XIC slice
    (same array `select_quantifiable_fragments_by_avg_corr` was run on).
    ``selected_indices``: fragment indices (into ``xic_arrays``'s 2nd axis)
    already chosen by shape correlation.

    Steps (see module caller for the full per-precursor pipeline):
    1. ``area[r, k] = sum_t xic_arrays[r, selected_indices[k], t]`` (fixed window).
    2. Per run ``r``, over its positive-finite areas only: run-center
       ``C_r = median_k log2(area[r, k] + epsilon)``, giving the
       precursor-abundance-invariant ``R[r, k] = log2(area[r,k]+eps) - C_r``
       (skipped entirely -- kept NaN -- for runs with < 2 such fragments,
       since a median of 0-1 points can't be trusted).
    3. Per fragment ``k``, over runs where ``R[r, k]`` is defined: reference
       ``B_k = median_r R[r, k]``, residual ``E[r, k] = R[r, k] - B_k``, and
       robust scale ``1.4826 * median_r |E[r, k]|``. Fragments observed in
       fewer than `min_interference_runs` such runs are never flagged (not
       enough evidence).
    4. Fragment ``k`` is interference (dropped in *every* run for this
       precursor, not just the offending run -- see module docstring) if any
       ``abs(E[r, k]) > max(interference_min_log2_fold, interference_z_threshold
       * scale_k)`` -- two-sided: both unusually *high* and unusually *low*
       run-specific relative fragment intensities are treated as
       interference. `interference_min_log2_fold` is an absolute log2-fold
       threshold, independent of residual direction. A fragment that is
       consistently high or low across *every* run is never flagged, since
       that's exactly what the per-fragment reference ``B_k`` absorbs.

    Returns the surviving fragment indices (subset of `selected_indices`, in
    the same order), or an empty array if fewer than `min_quant_fragments`
    survive -- the caller must then treat this precursor's quantity as
    missing rather than restoring any removed fragment.
    """
    n_runs = xic_arrays.shape[0]
    n_time = xic_arrays.shape[2]
    m = selected_indices.shape[0]

    # 1. fixed-window area per (run, selected fragment) -- unchanged window
    area = np.zeros((n_runs, m), dtype=np.float32)
    for r in range(n_runs):
        for k in range(m):
            f = selected_indices[k]
            s = 0.0
            for t in range(n_time):
                s += xic_arrays[r, f, t]
            area[r, k] = s

    # 2. run-centered relative intensity; NaN wherever undefined
    relative = np.full((n_runs, m), np.nan, dtype=np.float32)
    log_buf = np.empty(m, dtype=np.float32)
    for r in range(n_runs):
        cnt = 0
        for k in range(m):
            a = area[r, k]
            if np.isfinite(a) and a > 0.0:
                log_buf[cnt] = np.log2(a + epsilon)
                cnt += 1
        if cnt < 2:
            # too few positive-finite fragments to trust this run's median
            continue
        sorted_log = np.sort(log_buf[:cnt])
        mid = cnt // 2
        if cnt % 2 == 1:
            center = sorted_log[mid]
        else:
            center = (sorted_log[mid - 1] + sorted_log[mid]) * 0.5
        for k in range(m):
            a = area[r, k]
            if np.isfinite(a) and a > 0.0:
                relative[r, k] = np.log2(a + epsilon) - center

    # 3 & 4. per-fragment cross-run reference/residual/MAD -> interference flag
    interference = np.zeros(m, dtype=np.bool_)
    vals_buf = np.empty(n_runs, dtype=np.float32)
    for k in range(m):
        cnt = 0
        for r in range(n_runs):
            v = relative[r, k]
            if not np.isnan(v):
                vals_buf[cnt] = v
                cnt += 1
        if cnt < min_interference_runs:
            continue  # not enough cross-run evidence -> never flagged

        sorted_vals = np.sort(vals_buf[:cnt])
        mid = cnt // 2
        if cnt % 2 == 1:
            reference = sorted_vals[mid]
        else:
            reference = (sorted_vals[mid - 1] + sorted_vals[mid]) * 0.5

        abs_resid = np.empty(cnt, dtype=np.float32)
        for i in range(cnt):
            abs_resid[i] = abs(vals_buf[i] - reference)
        sorted_abs = np.sort(abs_resid)
        if cnt % 2 == 1:
            mad = sorted_abs[mid]
        else:
            mad = (sorted_abs[mid - 1] + sorted_abs[mid]) * 0.5
        threshold = max(
            interference_min_log2_fold, interference_z_threshold * 1.4826 * mad
        )

        for r in range(n_runs):
            v = relative[r, k]
            if np.isnan(v):
                continue
            residual = v - reference
            if abs(residual) > threshold:
                interference[k] = True
                break

    keep_count = 0
    for k in range(m):
        if not interference[k]:
            keep_count += 1

    if keep_count < min_quant_fragments:
        return np.empty(0, dtype=np.int32)

    final_indices = np.empty(keep_count, dtype=np.int32)
    pos = 0
    for k in range(m):
        if not interference[k]:
            final_indices[pos] = selected_indices[k]
            pos += 1
    return final_indices
