import numpy as np
import numba as nb

from delpi.utils.numeric import pearsonr


@nb.njit(nogil=True, fastmath=True, cache=True)
def _build_similarity_matrix(ref_indices, run_indices, xic_arrays):
    """
    Compute similarity matrix (ref x run).
    Uses .flatten() instead of .ravel() for safety with Numba sliced arrays.
    """
    m = ref_indices.size
    k = run_indices.size
    sim = np.empty((m, k), dtype=np.float32)

    for j in range(m):
        ref_idx = ref_indices[j]
        # [Safety] Use flatten() to ensure contiguous memory copy for pearsonr
        ref_xic = xic_arrays[ref_idx].flatten()

        for i in range(k):
            run_idx = run_indices[i]
            run_xic = xic_arrays[run_idx].flatten()

            # Pearson Correlation
            val = pearsonr(ref_xic, run_xic)
            sim[j, i] = val

    return sim


@nb.njit(nogil=True, fastmath=True, cache=True)
def _order_constrained_dp(sim_matrix, min_score=0.5, adapt_ratio=0.6):
    """
    Order-constrained DP with an adaptive per-row threshold.

    If a row has a strong best score, we raise the match threshold to keep that
    high-quality pairing instead of skipping it due to earlier low scores.
    """
    m, k = sim_matrix.shape
    dp = np.zeros((m + 1, k + 1), dtype=np.float32)
    prev = np.zeros(
        (m + 1, k + 1), dtype=np.int8
    )  # 0:Match, 1:Up(Skip Ref), 2:Left(Skip Run)

    row_max = np.empty(m, dtype=np.float32)
    for r in range(m):
        # 최대값이 매우 낮으면 그대로 min_score만 사용
        rmax = sim_matrix[r, 0]
        for c in range(1, k):
            if sim_matrix[r, c] > rmax:
                rmax = sim_matrix[r, c]
        row_max[r] = rmax

    for j in range(1, m + 1):
        for i in range(1, k + 1):
            score = sim_matrix[j - 1, i - 1]

            # Gap (skip) options
            val_up = dp[j - 1, i]  # Skip Ref
            val_left = dp[j, i - 1]  # Skip Run

            if val_up >= val_left:
                best_val = val_up
                best_dir = 1
            else:
                best_val = val_left
                best_dir = 2

            # Adaptive threshold: encourage keeping strong row peaks
            row_thr = min_score
            dyn_thr = row_max[j - 1] * adapt_ratio
            if dyn_thr > row_thr:
                row_thr = dyn_thr

            if score >= row_thr:
                val_match = dp[j - 1, i - 1] + score
                if val_match > best_val:
                    best_val = val_match
                    best_dir = 0

            dp[j, i] = best_val
            prev[j, i] = best_dir

    # ... (Backtracking 로직은 동일) ...
    match_run_idx = np.full(m, -1, dtype=np.int32)
    match_score = np.zeros(m, dtype=np.float32)

    j = m
    i = k
    while j > 0 and i > 0:
        move = prev[j, i]
        if move == 0:
            match_run_idx[j - 1] = i - 1
            match_score[j - 1] = sim_matrix[j - 1, i - 1]
            j -= 1
            i -= 1
        elif move == 1:
            j -= 1
        else:
            i -= 1

    return match_run_idx, match_score


@nb.njit(nogil=True, fastmath=True, cache=True)
def align_peptide_multi_spectra_matches(
    num_runs: int,
    ref_run_index: int,
    run_index_arr: np.ndarray,
    rt_arr: np.ndarray,
    xic_arr: np.ndarray,
):
    """
    Main function: RT-aware alignment and Voting
    """
    if run_index_arr.size == 1:
        # Only one PmSM available
        return (np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32))

    # 1. Prepare Reference PmSMs (Sorted by RT)
    ref_mask = run_index_arr == ref_run_index
    ref_indices = np.where(ref_mask)[0]

    if ref_indices.size == 0:
        return (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32))

    # Sort Ref by RT
    ref_order = np.argsort(rt_arr[ref_indices])
    ref_indices = ref_indices[ref_order]

    m = ref_indices.size

    # 2. Storage for Voting
    vote_count = np.zeros(m, dtype=np.int32)
    vote_score = np.zeros(m, dtype=np.float32)

    # Store alignment results to avoid re-calculation
    # per_run_match_to_ref[run_id, ref_pmsm_idx] = matched_run_pmsm_idx
    per_run_match_to_ref = np.full((num_runs, m), -1, dtype=np.int32)
    per_run_match_score = np.zeros((num_runs, m), dtype=np.float32)
    # Fallback: best-scoring PmSM per run/ref even when DP prunes the match
    per_run_best_to_ref = np.full((num_runs, m), -1, dtype=np.int32)
    per_run_best_score = np.full((num_runs, m), -1.0, dtype=np.float32)

    # 3. Align Each Run to Reference
    for run_id in range(num_runs):
        if run_id == ref_run_index:
            continue

        # Get Run PmSMs and Sort by RT
        run_indices_local = np.where(run_index_arr == run_id)[0]
        if run_indices_local.size == 0:
            continue

        run_order = np.argsort(rt_arr[run_indices_local])
        run_indices_sorted = run_indices_local[run_order]

        # Build Similarity Matrix
        sim_matrix = _build_similarity_matrix(ref_indices, run_indices_sorted, xic_arr)

        # Track the highest similarity per ref PmSM in this run for fallback
        for j in range(m):
            best_rel = -1
            best_val = -1.0
            for i in range(sim_matrix.shape[1]):
                val = sim_matrix[j, i]
                if val > best_val:
                    best_val = val
                    best_rel = i

            if best_rel >= 0:
                per_run_best_to_ref[run_id, j] = run_indices_sorted[best_rel]
                per_run_best_score[run_id, j] = best_val

        # Run DP Alignment (adaptive threshold keeps strong matches)
        matched_run_rel_idx, matched_scores = _order_constrained_dp(
            sim_matrix, min_score=0.5, adapt_ratio=0.8
        )

        # Collect Votes
        for j in range(m):
            rel_idx = matched_run_rel_idx[j]
            if rel_idx >= 0:
                # Valid match found
                real_run_idx = run_indices_sorted[rel_idx]

                vote_count[j] += 1
                vote_score[j] += matched_scores[j]

                per_run_match_to_ref[run_id, j] = real_run_idx
                per_run_match_score[run_id, j] = matched_scores[j]

    # 4. Select Representative Reference PmSM
    # Criteria: Most votes -> Highest cumulative score
    best_rep_idx = -1
    max_votes = -1
    max_score = -1.0

    for j in range(m):
        votes = vote_count[j]
        score = vote_score[j]

        if votes > max_votes:
            max_votes = votes
            max_score = score
            best_rep_idx = j
        elif votes == max_votes:
            if score > max_score:
                max_score = score
                best_rep_idx = j

    # 5. Output Result Formatting
    selected_indices = np.full(num_runs, -1, dtype=np.int32)
    selected_similarity = np.full(num_runs, -1.0, dtype=np.float32)

    # Set Reference Run Selection
    selected_indices[ref_run_index] = ref_indices[best_rep_idx]
    selected_similarity[ref_run_index] = 1.0

    # Set Other Runs Selection
    for run_id in range(num_runs):
        if run_id == ref_run_index:
            continue

        matched_idx = per_run_match_to_ref[run_id, best_rep_idx]
        if matched_idx != -1:
            selected_indices[run_id] = matched_idx
            selected_similarity[run_id] = per_run_match_score[run_id, best_rep_idx]
        else:
            # No DP match; fall back to the best-correlated PmSM in this run
            fallback_idx = per_run_best_to_ref[run_id, best_rep_idx]
            if fallback_idx != -1:
                selected_indices[run_id] = fallback_idx
                selected_similarity[run_id] = per_run_best_score[run_id, best_rep_idx]

    mask = selected_indices >= 0
    return (selected_indices[mask], selected_similarity[mask])
