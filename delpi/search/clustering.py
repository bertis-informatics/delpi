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
    """Assigns each PMSM to a cluster using greedy single-link criterion.

    Returns
    -------
    cluster_id_arr : int32[::1] length N - global cluster id per PMSM
    rep_idx_arr    : int32[::1] length *C* - row index of the representative
                     PMSM for each cluster, where *C* is the number of clusters
    """
    N = frame_num_arr.shape[0]

    # Output buffers (worst‑case N clusters → allocate length N)
    cluster_id_arr = np.empty(N, dtype=np.int32)
    rep_idx_arr = np.empty(N, dtype=np.int32)

    # Sort PMSMs by frame so that same‑frame rows are contiguous
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
def cluster_matches(
    frame_index_arr: np.ndarray,  # (N,)
    peak_index_arr: np.ndarray,  # (N, 128)
    max_frame_diff: int,
    jaccard_thres: float = 0.6,
) -> np.ndarray:
    """Assign each PMSM to a cluster using greedy single-link criterion.

    Two PMSMs are put in the *same* cluster if **both** conditions hold:
    1. ``abs(frame_num[i] - frame_num[rep]) ≤ 1``
    2. Jaccard similarity between their peak sets ≥ ``jaccard_thres``.

    The algorithm processes PMSMs sorted by ``frame_num`` so that candidate
    representative scans are encountered earlier; all heavy work is inside
    this nopython JIT kernel (no Python objects).

    Returns
    -------
    cluster_id_arr : int32[::1] length N – cluster id for each PMSM
    rep_idx_arr    : int32[::1] length *C* – row index of the representative
                     PMSM for each cluster (*C* = number of clusters)
    """
    N = frame_index_arr.shape[0]

    # Output buffers (maximum N clusters → allocate N)
    cluster_id_arr = np.empty(N, dtype=np.int32)
    rep_idx_arr = np.empty(N, dtype=np.int32)
    rep_frame_arr = np.empty(N, dtype=np.int32)  # frame number of each rep

    # Process PMSMs in ascending frame order for locality
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
