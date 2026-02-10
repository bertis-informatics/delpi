import numpy as np
import numba as nb

from delpi.utils.numeric import rowwise_pearsonr, pearsonr
from delpi.search.dia.peak_token import (
    EXP_IS_PRECURSOR_IDX,
    EXP_ISOTOPE_INDEX_IDX,
    EXP_MS_LEVEL_IDX,
    EXP_TIME_INDEX_IDX,
    EXP_AB_IDX,
)
from delpi.constants import RT_WINDOW_LEN, MAX_FRAGMENTS

MAX_THEO_INDEX = MAX_FRAGMENTS - 1


@nb.njit(parallel=True, cache=True)
def get_ms1_area(x_exp: np.ndarray, ms1_scale_arr: np.ndarray):

    N, M = x_exp.shape[:2]
    quant_arr = np.full(N, np.nan, dtype=np.float32)

    for i in nb.prange(N):
        x_arr = x_exp[i]
        scale = ms1_scale_arr[i]
        if scale <= 0:
            continue
        has_ms1_peak = False
        y = np.zeros(RT_WINDOW_LEN, dtype=np.float32)
        for j in range(M):
            t = nb.int8(x_arr[j, EXP_TIME_INDEX_IDX])
            if (x_arr[j, EXP_IS_PRECURSOR_IDX] > 0) and (
                x_arr[j, EXP_MS_LEVEL_IDX] == 1
            ):
                y[t] += x_arr[j, EXP_AB_IDX]
                has_ms1_peak = True

        if has_ms1_peak:
            y *= scale
            quant_arr[i] = np.sum(y)
            # quant_arr[i] = np.trapz(y)

    return quant_arr


@nb.njit(parallel=True, cache=True)
def get_ms1_area_dda(x_exp: np.ndarray, ms1_scale_arr: np.ndarray):
    N, M = x_exp.shape[:2]
    quant_arr = np.full(N, np.nan, dtype=np.float32)

    for i in nb.prange(x_exp.shape[0]):
        x_arr = x_exp[i]
        scale = ms1_scale_arr[i]
        if scale <= 0:
            continue
        # frame_idx = frame_index_arr[i]
        has_ms1_peak = False
        y = np.zeros(RT_WINDOW_LEN, dtype=np.float32)

        for j in range(M):
            t = nb.int8(x_arr[j, EXP_TIME_INDEX_IDX])
            if (x_arr[j, EXP_IS_PRECURSOR_IDX] > 0) and (
                x_arr[j, EXP_MS_LEVEL_IDX] == 1
            ):
                y[t] += x_arr[j, EXP_AB_IDX]
                has_ms1_peak = True

        if has_ms1_peak:
            y *= scale
            quant_arr[i] = np.sum(y)
            # x = ms1_rt_arr[frame_idx - xic_half_len : frame_idx + xic_half_len + 1]
            # quant_arr[i] = np.trapz(y)  # , x)

    return quant_arr


@nb.njit(cache=True)
def _nb_tri_index(i: int, j: int, n: int) -> int:
    # offset for first index i
    return (i * (2 * n - i - 1)) // 2 + (j - i - 1)


@nb.njit(cache=True)
def _nb_build_L_b(
    n_runs: int,
    pep_idx: np.ndarray,
    run_idx: np.ndarray,
    logI: np.ndarray,
):
    n_pairs = n_runs * (n_runs - 1) // 2
    counts = np.zeros(n_pairs, dtype=np.int64)

    # 1st pass: counts per pair
    N = pep_idx.shape[0]
    start = 0
    while start < N:
        pid = pep_idx[start]
        end = start + 1
        while end < N and pep_idx[end] == pid:
            end += 1
        m = end - start
        if m >= 2:
            for a in range(m - 1):
                ia = run_idx[start + a]
                la = logI[start + a]
                for b in range(a + 1, m):
                    ib = run_idx[start + b]
                    r = la - logI[start + b]
                    i = ia
                    j = ib
                    if i > j:
                        t = i
                        i = j
                        j = t
                        r = -r
                    p = _nb_tri_index(i, j, n_runs)
                    counts[p] += 1
        start = end

    offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    csum = 0
    for k in range(n_pairs):
        csum += counts[k]
        offsets[k + 1] = csum
    flat = np.empty(offsets[n_pairs], dtype=np.float64)
    write = np.zeros(n_pairs, dtype=np.int64)

    # 2nd pass: fill ratios
    start = 0
    while start < N:
        pid = pep_idx[start]
        end = start + 1
        while end < N and pep_idx[end] == pid:
            end += 1
        m = end - start
        if m >= 2:
            for a in range(m - 1):
                ia = run_idx[start + a]
                la = logI[start + a]
                for b in range(a + 1, m):
                    ib = run_idx[start + b]
                    r = la - logI[start + b]
                    i = ia
                    j = ib
                    if i > j:
                        t = i
                        i = j
                        j = t
                        r = -r
                    p = _nb_tri_index(i, j, n_runs)
                    pos = offsets[p] + write[p]
                    flat[pos] = r
                    write[p] += 1
        start = end

    # 3rd pass: median and accumulate L, b
    L = np.zeros((n_runs, n_runs), dtype=np.float64)
    b = np.zeros(n_runs, dtype=np.float64)

    p = 0
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            s = offsets[p]
            e = offsets[p + 1]
            cnt = e - s
            if cnt > 0:
                # sort the slice [s:e] with insertion sort (buckets are usually small)
                for x in range(s + 1, e):
                    key = flat[x]
                    y = x - 1
                    while y >= s and flat[y] > key:
                        flat[y + 1] = flat[y]
                        y -= 1
                    flat[y + 1] = key
                if cnt & 1:
                    med = float(flat[s + cnt // 2])
                else:
                    med = 0.5 * float(flat[s + cnt // 2 - 1] + flat[s + cnt // 2])
                w = float(cnt)
                L[i, i] += w
                L[j, j] += w
                L[i, j] -= w
                L[j, i] -= w
                b[i] += w * med
                b[j] -= w * med
            p += 1

    return L, b


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
    min_fragments=3,
    max_fragments=9,
    corr_thresh=0.5,
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
        min_fragments (int, optional): minimum number of fragments to select. Defaults to 3.
        max_fragments (int, optional): maximum number of fragments to select. Defaults to 6.
        corr_thresh (float, optional): minimum correlation threshold. Defaults to 0.5.
        cube_corr (bool, optional): whether to cube correlations. Defaults to False.
        rep_type (int, optional): 0 for consensus, 1 for representative. Defaults to 0.

    Returns:
        numpy.array: Indices of selected fragments sorted by average correlation (descending)
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

    # Return early if not enough valid fragments
    if valid_len == 0:
        # logically this should not happen
        return np.arange(n_frags, dtype=np.int32)

    if valid_len <= min_fragments:
        return valid_indices[:valid_len]

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

    if above_threshold_count >= min_fragments:
        # We have enough fragments above threshold
        # Select up to max_fragments among those above threshold
        selected_count = min(above_threshold_count, max_fragments)
    else:
        # Not enough fragments above threshold
        # Select top min_fragments regardless of threshold
        selected_count = min(min_fragments, valid_len)

    # Build the selected indices array
    selected_indices = np.empty(selected_count, dtype=np.int32)
    for i in range(selected_count):
        selected_indices[i] = valid_indices[sorted_order[i]]

    return selected_indices


@nb.njit(nogil=True, fastmath=True, parallel=True, cache=True)
def perform_lfq(
    precursor_index_arr,
    precursor_stop_index_arr,
    all_xic_arr,
    min_fragments: int = 6,
    max_fragments: int = 9,
    corr_thresh: float = 0.8,
    rep_type: int = 0,
    cube_corr: bool = False,
):
    all_ab_arr = np.zeros(all_xic_arr.shape[0], dtype=np.float32)

    for i in nb.prange(precursor_index_arr.shape[0]):
        st = 0 if i == 0 else precursor_stop_index_arr[i - 1]
        ed = precursor_stop_index_arr[i]
        sub_xic_arr = all_xic_arr[st:ed]

        selected_indices = select_quantifiable_fragments_by_avg_corr(
            sub_xic_arr,
            min_fragments=min_fragments,
            max_fragments=max_fragments,
            corr_thresh=corr_thresh,
            cube_corr=cube_corr,
            rep_type=rep_type,
        )

        for j in range(sub_xic_arr.shape[0]):
            for k in selected_indices:
                all_ab_arr[st + j] += np.sum(sub_xic_arr[j, k, :])

    return all_ab_arr
