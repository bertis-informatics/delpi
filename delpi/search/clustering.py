import numpy as np
import numba as nb


@nb.njit(inline="always")
def calc_jaccard_index(row_a: np.ndarray, row_b: np.ndarray) -> float:
    """Compute Jaccard(A,B) ignoring padding (-1). No duplicate indices assumed."""
    inter = 0
    len_a = 0
    len_b = 0
    # Count valid peaks in B
    for j in range(row_b.shape[0]):
        if row_b[j] < 0:
            break
        len_b += 1

    # Count valid peaks in A and intersection with B
    for i in range(row_a.shape[0]):
        a = row_a[i]
        if a < 0:
            break
        len_a += 1
        for j in range(len_b):
            b = row_b[j]
            if a == b:
                inter += 1
                break

    union = len_a + len_b - inter
    return 0.0 if union == 0 else inter / union


@nb.njit(inline="always")
def _jaccard(row_a: np.ndarray, row_b: np.ndarray) -> float:
    """Compute Jaccard(A, B) ignoring padding (-1). No duplicate indices assumed."""
    inter = 0
    len_a = 0
    len_b = 0

    row_a = row_a[row_a > -1]
    row_b = row_b[row_b > -1]
    len_a = row_a.shape[0]
    len_b = row_b.shape[0]

    for i in range(len_a):
        a = row_a[i]
        for j in range(len_b):
            b = row_b[j]
            if a == b:
                inter += 1
                break
    union = len_a + len_b - inter
    return 0.0 if union == 0 else inter / union


@nb.njit(cache=True)
def cluster_psms(
    frame_num_arr: np.ndarray,  # (N,)
    peak_index_arr: np.ndarray,  # (N, 64)
    jaccard_thres: float = 0.6,
) -> np.ndarray:
    """Assigns each PmSM to a cluster using greedy single-link criterion.

    Returns
    -------
    cluster_id_arr : int32[::1] length N - global cluster id per PmSM
    rep_idx_arr    : int32[::1] length *C* - row index of the representative
                     PmSM for each cluster, where *C* is the number of clusters
    """
    N = frame_num_arr.shape[0]

    # Output buffers (worst‑case N clusters → allocate length N)
    cluster_id_arr = np.empty(N, dtype=np.int32)
    rep_idx_arr = np.empty(N, dtype=np.int32)

    # Sort PmSMs by frame so that same‑frame rows are contiguous
    order = np.argsort(frame_num_arr)

    current_frame = -1
    frame_cluster_start = 0  # first cluster id for the current frame
    cluster_count = 0  # running total clusters (global id)

    for o in range(N):
        idx = order[o]
        frame = frame_num_arr[idx]

        # New frame ⇒ reset per‑frame variables
        if frame != current_frame:
            current_frame = frame
            frame_cluster_start = cluster_count  # next new cluster id will start here

        # Attempt to place idx into an existing cluster of this frame
        assigned = False
        for cid in range(frame_cluster_start, cluster_count):
            rep_idx = rep_idx_arr[cid]
            if (
                calc_jaccard_index(peak_index_arr[idx], peak_index_arr[rep_idx])
                >= jaccard_thres
            ):
                cluster_id_arr[idx] = cid
                assigned = True
                break

        if not assigned:
            # create new cluster
            cluster_id_arr[idx] = cluster_count
            rep_idx_arr[cluster_count] = idx
            cluster_count += 1

    # Trim rep_idx_arr to actual number of clusters and return
    # return cluster_id_arr, rep_idx_arr[:cluster_count]
    return cluster_id_arr


@nb.njit(cache=True)
def cluster_matches_jaccard(
    frame_index_arr: np.ndarray,  # (N,)
    peak_index_arr: np.ndarray,  # (N, peak_capacity)
    max_frame_diff: int,
    jaccard_thres: float = 0.6,
) -> np.ndarray:
    """Legacy frame-ordered clustering, retained for replaying historical results.

    Two PmSMs are put in the *same* cluster if **both** conditions hold:
    1. ``abs(frame_num[i] - frame_num[rep]) ≤ 1``
    2. Jaccard similarity between their peak sets ≥ ``jaccard_thres``.

    The algorithm processes PmSMs sorted by ``frame_num`` so that candidate
    representative scans are encountered earlier; all heavy work is inside
    this nopython JIT kernel (no Python objects).

    Returns
    -------
    cluster_id_arr : int32[::1] length N – cluster id for each PmSM
    rep_idx_arr    : int32[::1] length *C* – row index of the representative
                     PmSM for each cluster (*C* = number of clusters)
    """
    N = frame_index_arr.shape[0]

    # Output buffers (maximum N clusters → allocate N)
    cluster_id_arr = np.empty(N, dtype=np.int32)
    rep_idx_arr = np.empty(N, dtype=np.int32)
    rep_frame_arr = np.empty(N, dtype=np.int32)  # frame number of each rep

    # Process PmSMs in ascending frame order for locality
    order = np.argsort(frame_index_arr)
    cluster_count = 0

    for o in range(N):
        idx = order[o]
        frame = frame_index_arr[idx]

        assigned = False
        # Iterate over *all* existing clusters whose representative's frame is
        # within ±1 of current frame.  Since reps are stored in the order they
        # were created (which follows ascending frame numbers), we can break
        # early once rep_frame < frame‑1.
        for cid in range(
            cluster_count - 1, -1, -1
        ):  # iterate backwards for early break
            rep_frame = rep_frame_arr[cid]
            diff = frame - rep_frame  # current frame is >= rep_frame due to sort
            if diff > max_frame_diff:
                break  # older reps are >1 frame apart → cannot match
            if diff < -max_frame_diff:
                continue  # would never happen due to sort, kept for clarity
            # frame difference ≤ 1 – test Jaccard
            rep_idx = rep_idx_arr[cid]
            if (
                calc_jaccard_index(peak_index_arr[idx], peak_index_arr[rep_idx])
                >= jaccard_thres
            ):
                cluster_id_arr[idx] = cid
                assigned = True
                break

        if not assigned:
            # create new cluster
            cluster_id_arr[idx] = cluster_count
            rep_idx_arr[cluster_count] = idx
            rep_frame_arr[cluster_count] = frame
            cluster_count += 1

    # Trim rep_idx_arr to actual number of clusters and return
    # return cluster_id_arr, rep_idx_arr[:cluster_count]
    return cluster_id_arr


@nb.njit(inline="always")
def _count_shared_peaks(row_a, size_a, row_b, size_b):
    """Intersection size of sorted, unique peak IDs, excluding padding."""
    a, b, shared = 0, 0, 0
    while a < size_a and b < size_b:
        if row_a[a] == row_b[b]:
            shared += 1
            a += 1
            b += 1
        elif row_a[a] < row_b[b]:
            a += 1
        else:
            b += 1
    return shared


@nb.njit(nogil=True, cache=True)
def cluster_matches(
    frame_index_arr: np.ndarray,
    peak_index_arr: np.ndarray,
    score_arr: np.ndarray,
    max_frame_diff: int,
    containment_thres: float = 0.8,
) -> np.ndarray:
    """Cluster PmSMs by score-ordered NMS using asymmetric peak containment.

    Process higher scores first (input order breaks ties). Each candidate A
    compares its unique nonnegative peak IDs with every retained representative
    B within ``max_frame_diff``. If ``|A & B| / |A| >= containment_thres``, assign
    A to the highest-scoring qualifying representative's cluster. Otherwise A
    becomes a new representative. Empty peak sets never suppress or get suppressed.

    Suppressed candidates cannot suppress another candidate. All input rows keep
    a cluster ID for downstream rescoring/assignment; only representatives take
    part in subsequent NMS comparisons. Cluster IDs are dense and run-local once
    the caller applies its offset. Target/decoy labels are never used.
    """
    n = frame_index_arr.shape[0]
    if peak_index_arr.shape[0] != n or score_arr.shape[0] != n:
        raise ValueError("Frames, peak rows and scores must have equal lengths")
    if max_frame_diff < 0:
        raise ValueError("max_frame_diff must be nonnegative")
    if not 0 < containment_thres <= 1:
        raise ValueError("containment_thres must be in (0, 1]")
    if not np.isfinite(score_arr).all():
        raise ValueError("Clustering scores must be finite")

    cluster_ids = np.empty(n, dtype=np.int32)
    if n == 0:
        return cluster_ids

    # Deduplicate once per candidate, not once per pairwise comparison.
    unique_peaks = np.empty_like(peak_index_arr)
    peak_counts = np.empty(n, dtype=np.int32)
    for i in range(n):
        peaks = np.unique(peak_index_arr[i][peak_index_arr[i] >= 0])
        peak_counts[i] = len(peaks)
        unique_peaks[i, : len(peaks)] = peaks

    score_order = np.argsort(-score_arr, kind="mergesort")
    frame_order = np.argsort(frame_index_arr)
    sorted_frames = frame_index_arr[frame_order]
    # Unvisited and suppressed rows both have rank n and cannot represent a cluster.
    representative_rank = np.full(n, n, dtype=np.int64)
    cluster_count = 0
    for rank, i in enumerate(score_order):
        frame = np.int64(frame_index_arr[i])
        start = np.searchsorted(sorted_frames, frame - max_frame_diff)
        stop = np.searchsorted(sorted_frames, frame + max_frame_diff, side="right")
        best, best_rank = -1, n
        if peak_counts[i] > 0:
            required = containment_thres * peak_counts[i]
            for pos in range(start, stop):
                j = frame_order[pos]
                if representative_rank[j] >= best_rank or peak_counts[j] < required:
                    continue
                shared = _count_shared_peaks(
                    unique_peaks[i], peak_counts[i], unique_peaks[j], peak_counts[j]
                )
                if shared / peak_counts[i] >= containment_thres:
                    best, best_rank = j, representative_rank[j]
        if best >= 0:
            cluster_ids[i] = cluster_ids[best]
        else:
            cluster_ids[i] = cluster_count
            representative_rank[i] = rank
            cluster_count += 1
    return cluster_ids
