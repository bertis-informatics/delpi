"""Peak-sharing clustering of PmSMs.

A cluster is a run-local competition group for PmSM assignment
(:mod:`delpi.search.pmsm_assignment` assigns at most one precursor to each
cluster). Each PmSM is represented by the set of experimental fragment peaks
it matched over its whole RT window. Two PmSMs are linked when they share at
least `PEAK_SHARING_THRES` of the smaller set's peaks, and clusters are the
connected components of that graph. Scores and target/decoy labels are never
used.
"""

import numba as nb
import numpy as np

# Two PmSMs are linked when |A & B| / min(|A|, |B|) >= PEAK_SHARING_THRES.
PEAK_SHARING_THRES = 0.5


def compact_peak_indices(peak_index_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert padded peak-index rows into a flat (counts, indices) layout.

    `peak_index_rows` is an ``(N, W)`` array in which each row holds its peak
    indices first, followed by ``-1`` padding. Returns the number of valid
    entries per row and all valid entries concatenated in row order.
    """
    valid = peak_index_rows >= 0
    return (
        valid.sum(axis=1).astype(np.int32),
        peak_index_rows[valid].astype(np.int32),
    )


@nb.njit(inline="always")
def _find_root(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@nb.njit(nogil=True, cache=True)
def cluster_by_peak_sharing(
    peak_count_arr: np.ndarray,
    peak_index_arr: np.ndarray,
    share_thres: float = PEAK_SHARING_THRES,
) -> np.ndarray:
    """Cluster PmSMs as connected components of the peak-sharing graph.

    Args:
        peak_count_arr: ``(N,)`` number of peak indices of each PmSM.
        peak_index_arr: the PmSMs' peak indices concatenated in row order
            (``sum(peak_count_arr)`` entries). Indices must refer to one
            shared peak table; duplicates within a PmSM are counted once.
        share_thres: PmSMs A and B are linked when
            ``|A & B| / min(|A|, |B|) >= share_thres``.

    Returns:
        ``(N,)`` int32 cluster ids, dense from 0 and numbered in order of
        each cluster's first row. PmSMs without peaks are singletons.
    """
    n = peak_count_arr.shape[0]
    if not 0.0 < share_thres <= 1.0:
        raise ValueError("share_thres must be in (0, 1]")
    row_ptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        row_ptr[i + 1] = row_ptr[i] + peak_count_arr[i]
    if row_ptr[n] != peak_index_arr.shape[0]:
        raise ValueError("peak_index_arr length must equal sum(peak_count_arr)")

    # unique, sorted peaks per PmSM
    peaks = np.empty(row_ptr[n], dtype=peak_index_arr.dtype)
    ptr = np.zeros(n + 1, dtype=np.int64)
    m = 0
    for i in range(n):
        row = np.sort(peak_index_arr[row_ptr[i] : row_ptr[i + 1]])
        for k in range(row.shape[0]):
            if k == 0 or row[k] != row[k - 1]:
                peaks[m] = row[k]
                m += 1
        ptr[i + 1] = m
    peaks = peaks[:m]

    # inverted index: entries grouped by peak; the stable sort keeps rows
    # ascending inside each group, so the PmSMs after an entry are all j > i
    order = np.argsort(peaks, kind="mergesort")
    owner = np.empty(m, dtype=np.int64)
    for i in range(n):
        owner[ptr[i] : ptr[i + 1]] = i
    sorted_owner = owner[order]
    position = np.empty(m, dtype=np.int64)  # entry -> position in `order`
    group_end = np.empty(m, dtype=np.int64)  # entry -> end of its peak group
    s = 0
    while s < m:
        e = s + 1
        while e < m and peaks[order[e]] == peaks[order[s]]:
            e += 1
        for t in range(s, e):
            position[order[t]] = t
            group_end[order[t]] = e
        s = e

    parent = np.arange(n)
    shared = np.zeros(n, dtype=np.int32)
    touched = np.empty(n, dtype=np.int64)
    for i in range(n):
        n_touched = 0
        for k in range(ptr[i], ptr[i + 1]):
            for t in range(position[k] + 1, group_end[k]):
                j = sorted_owner[t]
                if shared[j] == 0:
                    touched[n_touched] = j
                    n_touched += 1
                shared[j] += 1
        size_i = ptr[i + 1] - ptr[i]
        for k in range(n_touched):
            j = touched[k]
            if shared[j] / min(size_i, ptr[j + 1] - ptr[j]) >= share_thres:
                ri = _find_root(parent, i)
                rj = _find_root(parent, j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
            shared[j] = 0

    cluster_ids = np.empty(n, dtype=np.int32)
    root_label = np.full(n, -1, dtype=np.int32)
    n_clusters = 0
    for i in range(n):
        r = _find_root(parent, i)
        if root_label[r] < 0:
            root_label[r] = n_clusters
            n_clusters += 1
        cluster_ids[i] = root_label[r]
    return cluster_ids
