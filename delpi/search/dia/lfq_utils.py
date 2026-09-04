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
from delpi.constants import (
    RT_WINDOW_LEN,
    RT_WINDOW_RADIUS,
    MAX_FRAGMENTS,
    QUANT_FRAGMENTS,
)

MAX_THEO_INDEX = MAX_FRAGMENTS - 1


@nb.njit(parallel=True, cache=True)
def get_pmsm_median_intensity(
    x_exp: np.ndarray,
    x_rank: np.ndarray,
    ms2_scale_arr: np.ndarray,
    neighbor_window: int = 1,
    top_k: int = 6,
) -> np.ndarray:
    """Median intensity of matched, center-window, monoisotopic, top-
    ``QUANT_FRAGMENTS``-ranked fragment ion peaks, computed directly from
    ``x_exp``'s peak annotations -- no separate XIC array needed. Shared by
    DIA and DDA, since both build ``x_exp``/``x_rank`` with the same
    conventions (center == ``RT_WINDOW_RADIUS``; ``x_rank`` only set, and
    always >= 0, for fragment peaks -- see `_set_x_exp`).

    Restricting to ``0 <= x_rank < QUANT_FRAGMENTS`` keeps this consistent
    with `get_xic_array`, which only ever holds those same ranked channels.

    ``x_exp``'s AB column is normalized to a per-PmSM max of 1 (see
    `_set_x_exp`'s fragment-ion scaling), so the median computed from it must
    be rescaled by ``ms2_scale_arr`` (that same call's fragment-ion max ab,
    as returned by `get_x_exp`) to recover the original intensity scale --
    the same role ``ms1_scale_arr`` plays in `get_ms1_area`.
    """
    N, M = x_exp.shape[:2]
    out = np.empty(N, dtype=np.float32)

    for n in nb.prange(N):
        x_arr = x_exp[n]
        rank_arr = x_rank[n]
        values = np.empty(M, dtype=np.float32)
        k = 0
        for j in range(M):
            rank = rank_arr[j]
            if 0 <= rank < top_k and x_arr[j, EXP_ISOTOPE_INDEX_IDX] == 0:
                t = nb.int8(x_arr[j, EXP_TIME_INDEX_IDX])
                if abs(t - RT_WINDOW_RADIUS) <= neighbor_window:
                    values[k] = x_arr[j, EXP_AB_IDX]
                    k += 1

        if k == 0:
            out[n] = np.nan
            continue

        vals = np.sort(values[:k])
        mid = k // 2
        if k % 2 == 1:
            med = vals[mid]
        else:
            med = (vals[mid - 1] + vals[mid]) * 0.5
        # median of a positively-scaled sample == scale * median(sample),
        # so rescaling once at the end is equivalent to (and cheaper than)
        # rescaling every value before taking the median
        out[n] = med * ms2_scale_arr[n]

    return out


@nb.njit(parallel=True, cache=True)
def get_xic_array(
    x_exp: np.ndarray, x_rank: np.ndarray, ms2_scale_arr: np.ndarray
) -> np.ndarray:
    """Build a ``(QUANT_FRAGMENTS, RT_WINDOW_LEN)`` max-held fragment-intensity
    array per PmSM directly from ``x_exp``, channel-indexed by ``x_rank``
    (each matched fragment peak's theoretical-intensity rank, as set by
    `delpi.search.dia.peak_token._set_x_exp` -- 0 == the library's most
    intense predicted fragment for this precursor; -1 == not ranked).

    Rank, not peak position, is required downstream: cross-run MaxLFQ
    (`perform_lfq`) selects fragment channels once per precursor group and
    reuses the same channel indices for every run's row, so the same
    theoretical fragment must always land in the same channel.

    ``x_exp``'s AB column is normalized to a per-PmSM max of 1, so the result
    is rescaled by ``ms2_scale_arr`` (see `get_pmsm_median_intensity`) to
    recover absolute intensity -- max-hold then scale is equivalent to scale
    then max-hold for a positive per-row factor, so it's applied once per row
    after the max-hold loop.
    """
    N, M = x_exp.shape[:2]
    xic = np.zeros((N, QUANT_FRAGMENTS, RT_WINDOW_LEN), dtype=np.float32)

    for n in nb.prange(N):
        x_arr = x_exp[n]
        rank_arr = x_rank[n]
        for j in range(M):
            channel = rank_arr[j]
            if 0 <= channel < QUANT_FRAGMENTS and x_arr[j, EXP_ISOTOPE_INDEX_IDX] == 0:
                t = nb.int8(x_arr[j, EXP_TIME_INDEX_IDX])
                if 0 <= t < RT_WINDOW_LEN:
                    ab = x_arr[j, EXP_AB_IDX]
                    if xic[n, channel, t] < ab:
                        xic[n, channel, t] = ab
        xic[n] *= ms2_scale_arr[n]

    return xic


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
    min_ratio_count: int,
):
    n_pairs = n_runs * (n_runs - 1) // 2
    # n_pairs/offsets are bounded by n_runs^2 and per-pair ratio counts by
    # peptides-per-protein, both always well within int32 range
    counts = np.zeros(n_pairs, dtype=np.int32)

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
                    if ia == ib:
                        # defensively skip duplicate (protein, precursor, run) rows
                        continue
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

    offsets = np.zeros(n_pairs + 1, dtype=np.int32)
    csum = 0
    for k in range(n_pairs):
        csum += counts[k]
        offsets[k + 1] = csum
    flat = np.empty(offsets[n_pairs], dtype=np.float32)
    write = np.zeros(n_pairs, dtype=np.int32)

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
                    if ia == ib:
                        continue
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
    # run pairs with fewer than min_ratio_count shared-precursor ratios are
    # left disconnected (no L/b contribution) rather than weighted down, so
    # connectivity - not weight - reflects how well-supported a pair is.
    L = np.zeros((n_runs, n_runs), dtype=np.float32)
    b = np.zeros(n_runs, dtype=np.float32)

    p = 0
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            s = offsets[p]
            e = offsets[p + 1]
            cnt = e - s
            if cnt >= min_ratio_count:
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
                w = 1.0
                L[i, i] += w
                L[j, j] += w
                L[i, j] -= w
                L[j, i] -= w
                b[i] += w * med
                b[j] -= w * med
            p += 1

    return L, b


@nb.njit(cache=True)
def _nb_find_root(parent: np.ndarray, i: int) -> int:
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        nxt = parent[i]
        parent[i] = root
        i = nxt
    return root


@nb.njit(cache=True)
def _nb_union(parent: np.ndarray, i: int, j: int):
    ri = _nb_find_root(parent, i)
    rj = _nb_find_root(parent, j)
    if ri != rj:
        parent[ri] = rj


@nb.njit(cache=True)
def _nb_maxlfq_all_proteins(
    protein_idx: np.ndarray,
    peptide_idx: np.ndarray,
    run_idx: np.ndarray,
    logI: np.ndarray,
    intensity: np.ndarray,
    n_runs_total: int,
    min_peptides_per_protein: int,
    min_ratio_count: int,
):
    """
    Vectorized MaxLFQ over all proteins in one compiled pass.

    Inputs must already be sorted by (protein_idx, peptide_idx); run_idx is a
    dense global run code aligned with the other arrays. For each protein
    this reproduces `_maxlfq_one_protein`'s logic (run/peptide grouping via
    `_nb_build_L_b`, connected components, per-component gauge fix + solve +
    rescale), but without any per-protein Python/Polars round trip.

    protein_idx/peptide_idx/run_idx are expected as int32 and logI/intensity
    as float32 (dense codes/observed values never need 64-bit range or
    precision here); the per-protein L/b/solve buffers are float32 too, since
    MaxLFQ ratios don't need double precision.
    """
    N = protein_idx.shape[0]

    out_protein = np.empty(N, dtype=np.int32)
    out_run = np.empty(N, dtype=np.int32)
    out_abundance = np.empty(N, dtype=np.float32)
    out_count = 0

    # reused across proteins via a "last seen protein id" stamp, avoiding an
    # O(n_runs_total) reset per protein
    remap_stamp = np.full(n_runs_total, -1, dtype=np.int32)
    remap_local = np.empty(n_runs_total, dtype=np.int32)
    local_to_global = np.empty(n_runs_total, dtype=np.int32)
    intensity_by_run_local = np.zeros(n_runs_total, dtype=np.float32)
    run_local_buf = np.empty(N, dtype=np.int32)

    row = 0
    while row < N:
        prot = protein_idx[row]
        prot_start = row
        prot_end = row + 1
        while prot_end < N and protein_idx[prot_end] == prot:
            prot_end += 1

        # single pass: local run remap, per-run intensity totals, distinct peptide count
        k = 0
        n_pep = 0
        prev_pep = -1
        for i in range(prot_start, prot_end):
            g = run_idx[i]
            if remap_stamp[g] != prot:
                remap_stamp[g] = prot
                remap_local[g] = k
                local_to_global[k] = g
                intensity_by_run_local[k] = 0.0
                k += 1
            loc = remap_local[g]
            intensity_by_run_local[loc] += intensity[i]
            run_local_buf[i] = loc

            pep = peptide_idx[i]
            if pep != prev_pep:
                n_pep += 1
                prev_pep = pep

        if n_pep < min_peptides_per_protein:
            row = prot_end
            continue

        if k == 1:
            out_protein[out_count] = prot
            out_run[out_count] = local_to_global[0]
            out_abundance[out_count] = intensity_by_run_local[0]
            out_count += 1
            row = prot_end
            continue

        L, b = _nb_build_L_b(
            k,
            peptide_idx[prot_start:prot_end],
            run_local_buf[prot_start:prot_end],
            logI[prot_start:prot_end],
            min_ratio_count,
        )

        # connected components via union-find over L's off-diagonal pattern
        # (all bounded by k = local run count for this protein, so int32 is ample)
        comp_parent = np.empty(k, dtype=np.int32)
        for t in range(k):
            comp_parent[t] = t
        for i in range(k):
            for j in range(i + 1, k):
                if L[i, j] != 0.0:
                    _nb_union(comp_parent, i, j)

        comp_root = np.empty(k, dtype=np.int32)
        for i in range(k):
            comp_root[i] = _nb_find_root(comp_parent, i)

        # counting sort: group local run indices by component root
        comp_count = np.zeros(k, dtype=np.int32)
        for i in range(k):
            comp_count[comp_root[i]] += 1
        comp_offset = np.zeros(k + 1, dtype=np.int32)
        for r in range(k):
            comp_offset[r + 1] = comp_offset[r] + comp_count[r]
        comp_write = np.zeros(k, dtype=np.int32)
        comp_members = np.empty(k, dtype=np.int32)
        for i in range(k):
            r = comp_root[i]
            pos = comp_offset[r] + comp_write[r]
            comp_members[pos] = i
            comp_write[r] += 1

        for r in range(k):
            cnt = comp_count[r]
            if cnt == 0:
                continue
            s = comp_offset[r]

            if cnt == 1:
                idx0 = comp_members[s]
                out_protein[out_count] = prot
                out_run[out_count] = local_to_global[idx0]
                out_abundance[out_count] = intensity_by_run_local[idx0]
                out_count += 1
                continue

            L_sub = np.empty((cnt, cnt), dtype=np.float32)
            b_sub = np.empty(cnt, dtype=np.float32)
            for a in range(cnt):
                ia = comp_members[s + a]
                for c in range(cnt):
                    L_sub[a, c] = L[ia, comp_members[s + c]]
                b_sub[a] = b[ia]

            # gauge fixing within this component: x_sub[0] = 0
            L_sub[0, :] = 0.0
            L_sub[:, 0] = 0.0
            L_sub[0, 0] = 1.0
            b_sub[0] = 0.0

            x_sub = np.linalg.solve(L_sub, b_sub)

            # relative linear-scale profile, rescaled to this component's observed intensity total
            max_x = np.max(x_sub)
            relative = np.exp(x_sub - max_x)
            rel_sum = np.sum(relative)

            total_intensity = 0.0
            for a in range(cnt):
                total_intensity += intensity_by_run_local[comp_members[s + a]]

            for a in range(cnt):
                idxa = comp_members[s + a]
                out_protein[out_count] = prot
                out_run[out_count] = local_to_global[idxa]
                out_abundance[out_count] = relative[a] * total_intensity / rel_sum
                out_count += 1

        row = prot_end

    return out_protein[:out_count], out_run[:out_count], out_abundance[:out_count]


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


def _validate_perform_lfq_params(
    target_fragments: int,
    min_quant_fragments: int,
    max_fragments: int,
    corr_thresh: float,
    min_interference_runs: int,
    interference_min_log2_fold: float,
    interference_z_threshold: float,
    epsilon: float,
) -> None:
    if min_quant_fragments < 1:
        raise ValueError(
            f"min_quant_fragments must be >= 1, got {min_quant_fragments!r}"
        )
    if target_fragments < 1:
        raise ValueError(f"target_fragments must be >= 1, got {target_fragments!r}")
    if max_fragments < 1:
        raise ValueError(f"max_fragments must be >= 1, got {max_fragments!r}")
    if not (min_quant_fragments <= target_fragments <= max_fragments):
        raise ValueError(
            "must satisfy min_quant_fragments <= target_fragments <= max_fragments, "
            f"got min_quant_fragments={min_quant_fragments!r}, "
            f"target_fragments={target_fragments!r}, max_fragments={max_fragments!r}"
        )
    if not (0 <= corr_thresh <= 1):
        raise ValueError(f"corr_thresh must be in [0, 1], got {corr_thresh!r}")
    if min_interference_runs < 1:
        raise ValueError(
            f"min_interference_runs must be >= 1, got {min_interference_runs!r}"
        )
    if interference_min_log2_fold <= 0:
        raise ValueError(
            "interference_min_log2_fold must be > 0, got "
            f"{interference_min_log2_fold!r}"
        )
    if interference_z_threshold <= 0:
        raise ValueError(
            f"interference_z_threshold must be > 0, got {interference_z_threshold!r}"
        )
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon!r}")


@nb.njit(nogil=True, fastmath=True, parallel=True, cache=True)
def _perform_lfq_numba(
    precursor_index_arr,
    precursor_stop_index_arr,
    all_xic_arr,
    target_fragments,
    min_quant_fragments,
    max_fragments,
    corr_thresh,
    rep_type,
    cube_corr,
    min_interference_runs,
    interference_min_log2_fold,
    interference_z_threshold,
    epsilon,
):
    # NaN (not 0) marks precursors left unquantified because too few
    # fragments survived shape-correlation + interference filtering
    all_ab_arr = np.full(all_xic_arr.shape[0], np.nan, dtype=np.float32)

    for i in nb.prange(precursor_index_arr.shape[0]):
        st = 0 if i == 0 else precursor_stop_index_arr[i - 1]
        ed = precursor_stop_index_arr[i]
        sub_xic_arr = all_xic_arr[st:ed]

        selected_indices = select_quantifiable_fragments_by_avg_corr(
            sub_xic_arr,
            target_fragments=target_fragments,
            max_fragments=max_fragments,
            corr_thresh=corr_thresh,
            cube_corr=cube_corr,
            rep_type=rep_type,
        )

        final_indices = filter_fragments_by_cross_run_intensity(
            sub_xic_arr,
            selected_indices,
            min_quant_fragments,
            min_interference_runs,
            interference_min_log2_fold,
            interference_z_threshold,
            epsilon,
        )

        if final_indices.shape[0] == 0:
            # too few fragments survived interference removal -> leave NaN
            # for every run of this precursor, don't restore removed fragments
            continue

        for j in range(sub_xic_arr.shape[0]):
            total = 0.0
            for k in final_indices:
                total += np.sum(sub_xic_arr[j, k, :])
            all_ab_arr[st + j] = total

    return all_ab_arr


def perform_lfq(
    precursor_index_arr,
    precursor_stop_index_arr,
    all_xic_arr,
    target_fragments: int = 9,
    min_quant_fragments: int = 3,
    max_fragments: int = 12,
    corr_thresh: float = 0.9,
    rep_type: int = 0,
    cube_corr: bool = False,
    min_interference_runs: int = 3,
    interference_min_log2_fold: float = 2.0,
    interference_z_threshold: float = 4.0,
    epsilon: float = 1e-6,
):
    """Per-precursor cross-run MS2 fragment-area quantification.

    For every precursor group (delimited by `precursor_stop_index_arr` into
    `all_xic_arr`'s rows): pick fragments via
    `select_quantifiable_fragments_by_avg_corr` (aiming for `target_fragments`,
    bounded by `max_fragments`/`corr_thresh`), drop any of those flagged by
    `filter_fragments_by_cross_run_intensity` as cross-run intensity
    interference, then sum the survivors' fixed-window areas per run.

    `target_fragments` is only a *selection* target -- it never determines
    whether a precursor's quantity is reported. That's `min_quant_fragments`'s
    sole job: if fewer than `min_quant_fragments` fragments survive both
    filters, every run of that precursor gets ``ms2_quantity = NaN`` instead
    of quantifying from a shrunken (but still valid) fragment set.

    Raises ``ValueError`` (before running any Numba code) if
    `min_quant_fragments`/`target_fragments`/`max_fragments` aren't a
    consistent ``min_quant_fragments <= target_fragments <= max_fragments``
    chain, or if any other parameter is out of range.
    """
    _validate_perform_lfq_params(
        target_fragments,
        min_quant_fragments,
        max_fragments,
        corr_thresh,
        min_interference_runs,
        interference_min_log2_fold,
        interference_z_threshold,
        epsilon,
    )
    return _perform_lfq_numba(
        precursor_index_arr,
        precursor_stop_index_arr,
        all_xic_arr,
        target_fragments,
        min_quant_fragments,
        max_fragments,
        corr_thresh,
        rep_type,
        cube_corr,
        min_interference_runs,
        interference_min_log2_fold,
        interference_z_threshold,
        epsilon,
    )


@nb.njit(parallel=True, cache=True)
def pmsm_median_intensity(xic_array, neighbor_window=0):
    """Median of matched (>0), center-window fragment intensities per PmSM,
    computed directly from an already-materialized ``(N, M, T)`` xic_array
    (e.g. from `ResultsAggregator.get_xic_arrays`) -- already on absolute
    intensity scale, so no rescaling is needed here.
    """
    N, M, T = xic_array.shape
    center = T // 2
    out = np.empty(N, dtype=xic_array.dtype)
    for n in nb.prange(N):
        values = np.empty(M * (2 * neighbor_window + 1), dtype=xic_array.dtype)
        k = 0
        for m in range(M):
            for t in range(center - neighbor_window, center + neighbor_window + 1):
                v = xic_array[n, m, t]
                if v > 0:
                    values[k] = v
                    k += 1
        if k == 0:
            out[n] = np.nan
            continue
        vals = np.sort(values[:k])
        mid = k // 2
        if k % 2 == 1:
            out[n] = vals[mid]
        else:
            out[n] = (vals[mid - 1] + vals[mid]) * 0.5

    return out
