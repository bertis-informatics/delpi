import numpy as np
import numba as nb


@nb.njit(nogil=True, fastmath=True, cache=True)
def find_local_maxima(x: np.ndarray):

    # fork from https://github.com/scipy/scipy/blob/v1.15.1/scipy/signal/_peak_finding_utils.pyx

    """Find local maxima, including plateaus, in a 1D array.

    A maximum is one or more equal samples with a smaller sample on both sides.
    Returns uint32 index arrays (midpoints, left_edges, right_edges); plateau
    midpoints are rounded down.
    """
    # cdef:
    #     np.intp_t[::1] midpoints, left_edges, right_edges
    #     np.intp_t m, i, i_ahead, i_max

    # Preallocate, there can't be more maxima than half the size of `x`
    midpoints = np.empty(x.shape[0] // 2, dtype=np.uint32)
    left_edges = np.empty(x.shape[0] // 2, dtype=np.uint32)
    right_edges = np.empty(x.shape[0] // 2, dtype=np.uint32)
    m = 0  # Pointer to the end of valid area in allocated arrays

    i = 1  # Pointer to current sample, first one can't be maxima
    i_max = x.shape[0] - 1  # Last sample can't be maxima
    while i < i_max:
        if x[i - 1] < x[i]:
            i_ahead = i + 1  # Index to look ahead of current sample

            # Find next sample that is unequal to x[i]
            while i_ahead < i_max and x[i_ahead] == x[i]:
                i_ahead += 1

            # Maxima is found if next unequal sample is smaller than x[i]
            if x[i_ahead] < x[i]:
                left_edges[m] = i
                right_edges[m] = i_ahead - 1
                midpoints[m] = (left_edges[m] + right_edges[m]) // 2
                m += 1
                # Skip samples that can't be maximum
                i = i_ahead
        i += 1

    # Keep only valid part of array memory (resize is not supported in numba).
    # midpoints.resize(m, refcheck=False)
    # left_edges.resize(m, refcheck=False)
    # right_edges.resize(m, refcheck=False)
    midpoints = midpoints[:m]
    left_edges = left_edges[:m]
    right_edges = right_edges[:m]

    return midpoints, left_edges, right_edges


@nb.njit(nogil=True, fastmath=True, cache=True)
def cluster_peaks(all_peaks, dist_cutoff: int = 3, min_cluster_size: int = 2):

    all_peaks.sort()
    if len(all_peaks) < 2:
        return all_peaks

    clustered_peaks = np.empty(all_peaks.shape[0], dtype=all_peaks.dtype)
    i = 0  # cluster_count

    current_cluster = np.empty(all_peaks.shape[0], dtype=all_peaks.dtype)
    current_cluster[0] = all_peaks[0]
    j = 1  # current_count

    for peak in all_peaks[1:]:
        if peak - current_cluster[j - 1] <= dist_cutoff:
            current_cluster[j] = peak
            j += 1
        else:
            if j >= min_cluster_size:
                pk = int(np.median(current_cluster[:j]))
                clustered_peaks[i] = pk
                i += 1
            # peak group을 averaging하면 apex 위치가 아니게 됨
            # for k in range(0, j, dist_cutoff):
            #     clustered_peaks[i] = current_cluster[k]
            #     i += 1
            current_cluster[0] = peak
            j = 1

    # add last cluster
    pk = int(np.median(current_cluster[:j]))
    clustered_peaks[i] = pk
    i += 1

    return clustered_peaks[:i]


@nb.njit(nogil=True, fastmath=True, cache=True)
def cluster_peaks_with_weights(
    peak_groups: np.ndarray,
    peak_group_weights: np.ndarray,
    dist_cutoff: int = 2,
    min_cluster_size: int = 1,
    max_cluster_size: int = 3,
):
    """Cluster peak positions and keep the highest-weight peak of each cluster.

    In position order, a peak joins the current cluster if it is within
    dist_cutoff of the cluster's last peak and the cluster has fewer than
    max_cluster_size peaks. Clusters smaller than min_cluster_size are dropped.
    """
    if len(peak_groups) < 2:
        return peak_groups

    # sort by position
    ii = np.argsort(peak_groups)
    peak_groups = peak_groups[ii]
    peak_group_weights = peak_group_weights[ii]

    clustered_peaks = np.empty_like(peak_groups)
    i = 0  # cluster_count

    current_cluster = np.empty_like(peak_groups)
    current_cluster_weight = np.empty_like(peak_group_weights)
    current_cluster[0] = peak_groups[0]
    current_cluster_weight[0] = peak_group_weights[0]
    j = 1  # current_count

    for peak, w in zip(peak_groups[1:], peak_group_weights[1:]):
        if peak - current_cluster[j - 1] <= dist_cutoff and j < max_cluster_size:
            current_cluster[j] = peak
            current_cluster_weight[j] = w
            j += 1
        else:
            # Finalize current cluster
            if j >= min_cluster_size:
                max_weight_idx = np.argmax(current_cluster_weight[:j])
                clustered_peaks[i] = current_cluster[max_weight_idx]
                i += 1

            current_cluster[0] = peak
            current_cluster_weight[0] = w
            j = 1

    # Finalize last cluster
    if j >= min_cluster_size:
        max_weight_idx = np.argmax(current_cluster_weight[:j])
        clustered_peaks[i] = current_cluster[max_weight_idx]
        i += 1

    return clustered_peaks[:i]


@nb.njit(nogil=True, fastmath=True, cache=True)
def find_bases(chrom: np.ndarray, peaks: np.ndarray):
    """Peak bases in a 1D chromatogram, found by walking downhill from each
    peak index; returns (left_bases, right_bases) aligned with peaks."""
    n = len(chrom)
    left_bases = np.empty_like(peaks)
    right_bases = np.empty_like(peaks)

    for idx, p in enumerate(peaks):
        # 왼쪽 기반점: p에서 chrom[i-1] < chrom[i]인 동안 왼쪽으로 이동
        i = p
        while i > 0 and chrom[i - 1] < chrom[i]:
            i -= 1
        left_bases[idx] = i

        # 오른쪽 기반점: p에서 chrom[j+1] < chrom[j]인 동안 오른쪽으로 이동
        j = p
        while j < n - 1 and chrom[j + 1] < chrom[j]:
            j += 1
        right_bases[idx] = j

    return left_bases, right_bases


@nb.njit(nogil=True, cache=True)
def nms_peaks(peaks: np.ndarray, weights: np.ndarray, dist: int = 2):
    """Suppress peaks within `dist` of a stronger peak; return sorted positions.

    Equal weights retain input order (scan order in peak-group detection).
    In position order, every peak within `dist` is an array neighbour, so each
    peak checks only those: O(K log K + K * dist) instead of O(K^2).
    """
    n = peaks.shape[0]
    if n < 2:
        return peaks
    order = np.argsort(-weights, kind="mergesort")
    is_sorted = True
    for i in range(1, n):
        if peaks[i] < peaks[i - 1]:
            is_sorted = False
            break
    if is_sorted:  # scan-sorted in peak-group detection
        pos = peaks.copy()
    else:
        by_pos = np.argsort(peaks, kind="mergesort")
        pos = peaks[by_pos]
        rank = np.empty(n, dtype=np.int64)
        rank[by_pos] = np.arange(n)
        order = rank[order]
    keep = np.zeros(n, dtype=np.bool_)
    for a in range(n):
        ia = order[a]
        p = np.int64(pos[ia])
        ok = True
        # Neighbours not visited yet still have keep == False.
        j = ia - 1
        while ok and j >= 0 and p - np.int64(pos[j]) <= dist:
            ok = not keep[j]
            j -= 1
        j = ia + 1
        while ok and j < n and np.int64(pos[j]) - p <= dist:
            ok = not keep[j]
            j += 1
        keep[ia] = ok
    return pos[keep]
