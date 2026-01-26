import numpy as np
import numba as nb

from delpi.utils.numeric import rowwise_pearsonr


@nb.njit(nogil=True, fastmath=True)
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


@nb.njit(nogil=True, fastmath=True)
def select_quantifiable_fragments(xic_arrays, min_fragments=3, corr_thresh=0.8):
    """
    Recursive Co-elution Elimination

    Args:
        xic_arrays (numpy.array): [n_runs, n_frags, n_time]
        min_fragments (int, optional): minimum number of fragments to select. Defaults to 3.
        corr_thresh (float, optional): correlation threshold for elimination. Defaults to 0.8.
    Returns:
        numpy.array: Indices of selected fragments
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

    active_indices = np.empty(n_frags, dtype=np.int64)
    active_len = 0
    for frag_idx in range(n_frags):
        if total_intensities[frag_idx] > epsilon:
            active_indices[active_len] = frag_idx
            active_len += 1

    # Return early if the remaining fragments are already below the requested count
    if active_len <= min_fragments:
        return active_indices[:active_len]

    score_accumulator = np.empty(n_frags, dtype=np.float32)
    # work_xics = np.empty((n_frags, n_time), dtype=np.float32)

    # 2. Recursive Elimination Loop
    while active_len > min_fragments:

        # Calculate average scores across runs for each active fragment
        # score_accumulator: Accumulates scores in the order of current active_indices
        score_accumulator[:] = 0.0

        # iterate over runs
        for run_idx in range(n_runs):
            # Extract the current run's XICs for the active fragments (shape: [k, T])
            current_xics = xic_arrays[run_idx, active_indices[:active_len], :]

            # A. Build a robust consensus (median) so outliers do not poison the reference
            consensus = get_consensus_xic(current_xics)

            # B. Compute correlations between each fragment and the consensus
            run_corrs = rowwise_pearsonr(current_xics, consensus)

            # Accumulate the scores, cubing each correlation so single-run outliers
            # (e.g., 0.1, 0.9, 0.9) are penalized more than consistently good pairs
            for row_idx in range(active_len):
                corr = run_corrs[row_idx]
                score_accumulator[row_idx] += corr

        # average correlation
        for i in range(active_len):
            score_accumulator[i] = score_accumulator[i] / n_runs

        # Find the fragment with the lowest (worst) score
        min_idx = 0
        min_score = score_accumulator[0]
        for i in range(1, active_len):
            score = score_accumulator[i]
            if score < min_score:
                min_score = score
                min_idx = i

        # 3. Elimination Decision
        if min_score < corr_thresh:
            # Remove fragment if below threshold (pop from active list)
            for i in range(min_idx, active_len - 1):
                active_indices[i] = active_indices[i + 1]
            active_len -= 1
        else:
            # If the worst fragment passes the threshold, stop (all good)
            break

    selected_indices = active_indices[:active_len]
    # scores = score_accumulator[selected_indices]
    # return selected_indices, scores
    return selected_indices


@nb.njit(nogil=True, fastmath=True)
def quantify_fragments(
    xic_arrays: np.ndarray,
    min_fragments: int = 3,
    corr_thresh: float = 0.8,
) -> np.ndarray:
    """Select quantifiable fragments based on co-elution similarity.

    Args:
        xic_arrays (numpy.array): [n_runs, n_frags, n_time]
        min_fragments (int, optional): minimum number of fragments to select. Defaults to 3.
        corr_thresh (float, optional): correlation threshold for elimination. Defaults to 0.8.

    Returns:
        numpy.array: Indices of selected fragments
    """

    selected_indices = select_quantifiable_fragments(
        xic_arrays, min_fragments=min_fragments, corr_thresh=corr_thresh
    )
    quantified_abundance = np.empty(xic_arrays.shape[0], dtype=np.float32)

    for run_idx, xic_array in enumerate(xic_arrays):
        # xic = np.sum(xic_array[selected_indices, :] * scores[:, None], axis=0)
        xic = np.sum(xic_array[selected_indices, :], axis=0)
        quantified_abundance[run_idx] = np.trapz(xic)

    return quantified_abundance
