import numpy as np
import numba as nb


from delpi.utils.numeric import corrcoef, rowwise_pearsonr
from delpi.search.dia.peak_token import (
    EXP_IS_PRECURSOR_IDX,
    EXP_ISOTOPE_INDEX_IDX,
    EXP_MS_LEVEL_IDX,
    EXP_TIME_INDEX_IDX,
    EXP_AB_IDX,
)
from delpi.search.dia.peak_token import QUANT_FRAGMENTS

# Constants
MAX_THEO_INDEX = 15
XIC_LEN = 7


@nb.njit(cache=True, parallel=True)
def get_apex_median_intensity(quant_ab_arr, quant_time_index_arr) -> np.ndarray:
    n = quant_ab_arr.shape[0]
    apex_median_intensity_arr = np.empty(n, dtype=np.float32)
    for i in nb.prange(n):
        ab_arr = quant_ab_arr[i]
        time_arr = quant_time_index_arr[i]
        mask = (time_arr >= 3) & (time_arr <= 5) & (ab_arr > 0)
        apex_ab_arr = ab_arr[mask]
        apex_median_intensity_arr[i] = (
            np.median(apex_ab_arr) if len(apex_ab_arr) > 0 else 0.0
        )

    return apex_median_intensity_arr


@nb.njit(parallel=True, cache=True)
def get_ms1_area(
    ms1_rt_arr: np.ndarray,
    frame_index_arr: np.ndarray,
    x_exp: np.ndarray,
    ms1_scale_arr: np.ndarray,
):
    xic_half_len = XIC_LEN // 2
    N, M = x_exp.shape[:2]
    quant_arr = np.full(N, np.nan, dtype=np.float32)

    for i in nb.prange(x_exp.shape[0]):
        x_arr = x_exp[i]
        scale = ms1_scale_arr[i]
        frame_idx = frame_index_arr[i]
        if scale <= 0:
            continue

        has_ms1_peak = False
        y = np.zeros(XIC_LEN, dtype=np.float32)
        for j in range(M):
            t = nb.int8(x_arr[j, EXP_TIME_INDEX_IDX])
            if (
                (x_arr[j, EXP_IS_PRECURSOR_IDX] > 0)
                and (x_arr[j, EXP_MS_LEVEL_IDX] == 1)
                and (t > 0)
                and (t < 8)
            ):
                y[t - 1] += x_arr[j, EXP_AB_IDX]
                has_ms1_peak = True

        if has_ms1_peak:
            y *= scale
            x = ms1_rt_arr[frame_idx - xic_half_len : frame_idx + xic_half_len + 1]
            quant_arr[i] = np.trapz(y, x)

    return quant_arr


@nb.njit(inline="always")
def get_xic_arr(ab_arr, theo_arr, time_arr):
    # time index is in [1, 7]
    out_arr = np.zeros((QUANT_FRAGMENTS, XIC_LEN), dtype=np.float32)
    for i, t, ab in zip(theo_arr, time_arr, ab_arr):
        if i < 0:
            break
        out_arr[MAX_THEO_INDEX - i, t - 1] = max(out_arr[MAX_THEO_INDEX - i, t - 1], ab)
    return out_arr


@nb.njit(cache=True)
def percentile_numba(arr, q):
    """Numba-friendly percentile for 2D array (row-wise)."""
    m, n = arr.shape
    out = np.empty(m, dtype=np.float32)
    q_index = (q / 100.0) * (n - 1)
    for i in range(m):
        row = np.sort(arr[i])
        k = int(q_index)
        f = q_index - k
        if k + 1 < n:
            out[i] = row[k] * (1 - f) + row[k + 1] * f
        else:
            out[i] = row[k]
    return out


@nb.njit(cache=True)
def get_rep_xic(xic_arr):
    """
    xic_arr: [n_frag, n_time]
    return: reference XIC (mean of normalized fragments)
    """
    n_frag, n_time = xic_arr.shape

    # 1) 유효 fragment (모든 intensity가 0인 fragment 제외)
    valid_mask = np.zeros(n_frag, dtype=np.bool_)
    for i in range(n_frag):
        if np.sum(xic_arr[i, :]) > 0.0:
            valid_mask[i] = True
    m = np.sum(valid_mask)

    if m == 0:
        # 모든 fragment가 0이면 0 벡터 반환
        return np.zeros(n_time, dtype=np.float32)

    valid_idx = np.empty(m, dtype=np.int32)
    idx = 0
    for i in range(n_frag):
        if valid_mask[i]:
            valid_idx[idx] = i
            idx += 1

    x_valid = xic_arr[valid_idx, :]  # [m, n_time]

    # 2) 95th percentile scaling
    scale = percentile_numba(x_valid, 95.0)  # [m]
    for i in range(m):
        if scale[i] < 1e-9:
            scale[i] = 1e-9

    # 3) normalize & mean
    ref = np.zeros(n_time, dtype=np.float32)
    for i in range(m):
        for t in range(n_time):
            ref[t] += x_valid[i, t] / scale[i]
    ref /= m

    return ref


@nb.njit(inline="always")
def snr(x):
    base = np.mean(np.partition(x, int(0.2 * len(x)))[: int(0.2 * len(x))]) + 1e-9
    return float(np.max(x) / base)


@nb.njit(parallel=True, cache=True)
def score_fragment_xics(quant_ab_arr, quant_theo_arr, quant_time_arr, eps=1e-9):
    frag_score_arr = np.full(
        (quant_ab_arr.shape[0], QUANT_FRAGMENTS), np.nan, dtype=np.float32
    )
    frag_intensity_arr = np.zeros(
        (quant_ab_arr.shape[0], QUANT_FRAGMENTS), dtype=np.float32
    )

    for i in nb.prange(frag_score_arr.shape[0]):
        ab_arr = quant_ab_arr[i]
        theo_arr = quant_theo_arr[i]
        time_arr = quant_time_arr[i]
        xic_arr = get_xic_arr(ab_arr, theo_arr, time_arr)
        rep_xic = get_rep_xic(xic_arr)

        intensity_arr = xic_arr[:, 2:-2].sum(axis=1)
        positive = intensity_arr[intensity_arr > 0]
        if positive.size == 0:
            continue

        intensity_scale = max(np.median(positive), 1e-9)
        intensity_arr /= intensity_scale
        frag_intensity_arr[i, :] = intensity_arr

        ii = np.where(intensity_arr > 0)[0]
        corr = rowwise_pearsonr(xic_arr[ii], rep_xic)
        frag_score_arr[i, ii] = corr**3

        # ii = np.where(np.sum(xic_arr, axis=1) > 0)[0]
        # if len(ii) > 1:
        #     corr = corrcoef(xic_arr[ii])
        #     j = np.argmax(np.sum(corr**3, axis=1))
        #     frag_score_arr[i, ii] = corr[j, :]

    return frag_score_arr, frag_intensity_arr


@nb.njit(parallel=True, cache=True)
def quant_fragment_xics(
    frame_to_rt_map_arr,
    quant_ab_arr,
    quant_theo_arr,
    quant_time_arr,
    selected_index_arr,
    frag_score_arr,
    frame_index_arr,
    win_index_arr,
):
    xic_half_len = XIC_LEN // 2
    N = selected_index_arr.shape[0]
    quant_arr = np.zeros(N, dtype=np.float32)

    for i in nb.prange(N):
        win_idx = win_index_arr[i]
        frame_idx = frame_index_arr[i]

        ab_arr = quant_ab_arr[i]
        theo_arr = quant_theo_arr[i]
        time_arr = quant_time_arr[i]
        selected_indices = selected_index_arr[i]
        xic_arr = get_xic_arr(ab_arr, theo_arr, time_arr)

        w = frag_score_arr[i, selected_indices]

        ws = w.sum()
        if ws < 1e-9:
            quant_arr[i] = 0.0
            continue

        w /= ws
        # y = (w[:, None] * xic_arr[selected_indices, :]).sum(axis=0)
        # y = xic_arr[selected_indices, :].sum(axis=0)
        y = np.zeros(XIC_LEN, dtype=np.float32)
        for j, k in enumerate(selected_indices):
            y[:] += w[j] * xic_arr[k]

        x = frame_to_rt_map_arr[
            win_idx, frame_idx - xic_half_len : frame_idx + xic_half_len + 1
        ]

        peak_area = np.trapz(y, x)
        quant_arr[i] = peak_area

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
        y = np.zeros(XIC_LEN, dtype=np.float32)

        for j in range(M):
            t = nb.int8(x_arr[j, EXP_TIME_INDEX_IDX])
            if (
                (x_arr[j, EXP_IS_PRECURSOR_IDX] > 0)
                and (x_arr[j, EXP_MS_LEVEL_IDX] == 1)
                and (t > 0)
                and (t < 8)
            ):
                y[t - 1] += x_arr[j, EXP_AB_IDX]
                has_ms1_peak = True

        if has_ms1_peak:
            y *= scale
            # x = ms1_rt_arr[frame_idx - xic_half_len : frame_idx + xic_half_len + 1]
            quant_arr[i] = np.trapz(y)  # , x)

    return quant_arr


@nb.njit(cache=True)
def _mad_log_no_nan_1d(x: np.ndarray, eps: float = 1e-6) -> float:
    """
    median(|log(x)-median(log(x))|)
    """
    n = x.size
    cnt = 0
    for i in range(n):
        if not np.isnan(x[i]):
            cnt += 1
    if cnt == 0:
        return np.nan

    buf = np.empty(cnt, dtype=np.float64)
    k = 0
    for i in range(n):
        if not np.isnan(x[i]):
            v = x[i]
            if v <= 0.0:
                v = eps
            buf[k] = np.log(v)
            k += 1

    med = float(np.median(buf))
    # |logI - med|
    for i in range(cnt):
        buf[i] = abs(buf[i] - med)
    return float(np.median(buf))


@nb.njit(parallel=True, cache=True)
def find_quantitative_fragments(
    score_matrix: np.ndarray,  # [P, R, F], correlation (NaN 허용)
    intensity_matrix: np.ndarray,  # [P, R, F], normalized apex intensity
    beta: float = 2.0,
    topk: int = 3,
):
    """
    반환:
      topk_idx: [P, K]  (precursor별 상위 fragment 인덱스, 내림차순)
      S       : [P, F]  (fragment 점수)
    """
    P, R, F = score_matrix.shape
    K = topk if topk < F else F

    S = np.zeros((P, F), dtype=np.float32)
    topk_idx = np.zeros((P, K), dtype=np.uint32)

    for p in nb.prange(P):
        for f in range(F):

            corr_slice = score_matrix[p, :, f]
            inten_slice = intensity_matrix[p, :, f]

            if np.all(inten_slice < 1e-6):
                # intensities are all zero across runs
                S[p, f] = 0.0
                continue

            mask = ~np.isnan(corr_slice)
            med_corr = np.median(corr_slice[mask]) if np.any(mask) else np.nan

            # MAD(log I)
            mad_logI = _mad_log_no_nan_1d(inten_slice)

            if np.isnan(med_corr) or np.isnan(mad_logI):
                S[p, f] = 0.0
            else:
                S[p, f] = med_corr * np.exp(-beta * mad_logI)

        topk_idx[p, :] = np.argsort(S[p])[-topk:]

    return topk_idx, S


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
