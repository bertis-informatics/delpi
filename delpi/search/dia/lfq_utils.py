import numpy as np
import numba as nb

from delpi.search.dia.lfq_frag_select import (
    select_quantifiable_fragments_by_avg_corr,
    filter_fragments_by_cross_run_intensity,
)
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
    quant_window_radius,
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

        if quant_window_radius is not None:
            center = sub_xic_arr.shape[2] // 2
            start = center - quant_window_radius
            stop = center + quant_window_radius + 1
            sub_xic_arr = sub_xic_arr[:, :, start:stop]

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
    quant_window_radius: int = None,
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
        quant_window_radius,
        epsilon,
    )
