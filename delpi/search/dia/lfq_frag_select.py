import numpy as np
import numba as nb

from delpi.utils.numeric import rowwise_pearsonr, pearsonr


@nb.njit(nogil=True, fastmath=True, cache=True)
def get_consensus_xic(xic_arr):
    """Per-time median of fragment XICs, each scaled by its 95th percentile.

    `xic_arr`: (n_frag, n_time). Returns zeros if no fragment has signal.
    """
    n_frag, n_time = xic_arr.shape
    epsilon = 1e-9
    norm_buf = np.empty((n_frag, n_time), dtype=np.float32)
    valid_cnt = 0

    for i in range(n_frag):
        row = xic_arr[i]
        if np.sum(row) < epsilon:
            continue

        sorted_row = np.sort(row)
        scale = sorted_row[int(0.95 * (n_time - 1))]

        if scale < epsilon:
            continue

        norm_buf[valid_cnt] = row / scale
        valid_cnt += 1

    if valid_cnt == 0:
        return np.zeros(n_time, dtype=np.float32)

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
    """Fragment XIC with the highest mean cubed correlation to the other fragments.

    Similar to DIA-NN's representative XIC. `xic_arr`: (n_frag, n_time); returns
    a copy of one row, or zeros if every fragment is all-zero.
    """
    n_frag, n_time = xic_arr.shape
    epsilon = 1e-9

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

            corr = pearsonr(xic_i, xic_j)
            total_corr += corr**3

        avg_corr[i] = total_corr / (valid_cnt - 1)

    best_idx = 0
    best_corr = avg_corr[0]

    for i in range(1, valid_cnt):
        if avg_corr[i] > best_corr:
            best_corr = avg_corr[i]
            best_idx = i

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
    """Select fragments by mean correlation with each run's reference XIC.

    `xic_arrays`: (n_runs, n_frags, n_time). Per run, each fragment's Pearson
    correlation with the consensus (`rep_type` 0) or representative (1) XIC is
    taken, optionally cubed with sign kept, and averaged over runs. If at least
    `target_fragments` reach `corr_thresh`, up to `max_fragments` of them are
    kept; otherwise the top `target_fragments` (at most `max_fragments`).
    `target_fragments` is not a reporting cutoff; that is `min_quant_fragments`
    in `filter_fragments_by_cross_run_intensity`.

    Returns fragment indices sorted by mean correlation (descending), empty if
    every fragment has near-zero total intensity.
    """
    n_runs, n_frags, n_time = xic_arrays.shape
    epsilon = 1e-6

    # 1. drop fragments with near-zero total intensity over all runs
    total_intensities = np.zeros(n_frags, dtype=np.float32)
    for frag_idx in range(n_frags):
        total = 0.0
        for run_idx in range(n_runs):
            for t in range(n_time):
                total += xic_arrays[run_idx, frag_idx, t]
        total_intensities[frag_idx] = total

    valid_indices = np.empty(n_frags, dtype=np.int32)
    valid_len = 0
    for frag_idx in range(n_frags):
        if total_intensities[frag_idx] > epsilon:
            valid_indices[valid_len] = frag_idx
            valid_len += 1

    if valid_len == 0:
        return np.empty(0, dtype=np.int32)

    # 2. mean correlation with each run's reference XIC
    avg_correlations = np.zeros(valid_len, dtype=np.float32)

    for run_idx in range(n_runs):
        current_xics = xic_arrays[run_idx, valid_indices[:valid_len], :]

        if rep_type == 0:
            consensus = get_consensus_xic(current_xics)
        else:
            consensus = get_representative_xic(current_xics)

        run_corrs = rowwise_pearsonr(current_xics, consensus)

        for i in range(valid_len):
            corr = run_corrs[i]
            if cube_corr:
                if corr < 0:
                    corr = -((-corr) ** 3)
                else:
                    corr = corr**3

            avg_correlations[i] += corr

    for i in range(valid_len):
        avg_correlations[i] = avg_correlations[i] / n_runs

    # 3. sort by mean correlation, descending
    sorted_order = np.argsort(-avg_correlations)

    # 4. count fragments at or above corr_thresh
    above_threshold_count = 0
    for i in range(valid_len):
        if avg_correlations[sorted_order[i]] >= corr_thresh:
            above_threshold_count += 1
        else:
            break  # sorted: no more pass

    if above_threshold_count >= target_fragments:
        selected_count = min(above_threshold_count, max_fragments)
    else:
        # too few pass: take the top target_fragments anyway
        selected_count = min(target_fragments, valid_len, max_fragments)

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
    """Drop fragments whose run-relative intensity is an outlier in some runs.

    Shape correlation is scale invariant, so a fragment can pass
    `select_quantifiable_fragments_by_avg_corr` while one run's intensity is
    distorted by interference or signal loss. Only run-centered relative
    intensities are compared, so neither a globally weak fragment nor a run
    with high or low precursor abundance is flagged, nor a fragment that is
    consistently high or low in every run (absorbed by ``B_k``).

    `xic_arrays` is the precursor's (n_runs, n_frags, n_time) XIC slice and
    `selected_indices` the fragments (axis 1) kept by shape correlation.

    1. ``area[r, k] = sum_t xic_arrays[r, selected_indices[k], t]``.
    2. Per run, over positive finite areas: ``C_r = median_k log2(area + eps)``
       and ``R[r, k] = log2(area[r, k] + eps) - C_r``. Runs with fewer than 2
       such fragments stay NaN.
    3. Per fragment, over runs where ``R`` is defined: ``B_k = median_r R[r, k]``,
       ``E[r, k] = R[r, k] - B_k`` and ``scale_k = 1.4826 * median_r |E[r, k]|``.
       Fragments seen in fewer than `min_interference_runs` runs are never
       flagged.
    4. Fragment ``k`` is interference if any ``|E[r, k]|`` exceeds
       ``max(interference_min_log2_fold, interference_z_threshold * scale_k)``
       (two-sided), and is then dropped in every run.

    Returns the surviving indices in `selected_indices` order, or an empty
    array if fewer than `min_quant_fragments` survive; the caller then leaves
    the precursor unquantified instead of restoring dropped fragments.
    """
    n_runs = xic_arrays.shape[0]
    n_time = xic_arrays.shape[2]
    m = selected_indices.shape[0]

    # 1. area per (run, selected fragment)
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

    # 3-4. per-fragment reference, MAD and interference flag
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
            continue  # too few runs: never flagged

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
