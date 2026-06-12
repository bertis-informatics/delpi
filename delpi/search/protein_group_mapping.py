from collections import defaultdict
from typing import Tuple

import numpy as np
import numba as nb
import polars as pl


def _build_csr(precursor_index_arr: np.ndarray, protein_index_arr: np.ndarray):
    """Build compressed-sparse-row (CSR) adjacency lists for the bipartite
    protein<->precursor graph.

    Proteins and precursors are remapped to dense local indices (0..P-1, 0..N-1).
    ``protein_ids``/``precursor_ids`` are the sorted unique original ids, so local
    index order follows ascending original index (this fixes the greedy
    tie-breaking deterministically to "smallest protein index wins").
    """
    protein_ids, prot_local = np.unique(protein_index_arr, return_inverse=True)
    precursor_ids, prec_local = np.unique(precursor_index_arr, return_inverse=True)
    prot_local = prot_local.ravel().astype(np.int64)
    prec_local = prec_local.ravel().astype(np.int64)

    num_proteins = len(protein_ids)
    num_precursors = len(precursor_ids)

    # protein -> precursors CSR
    order = np.argsort(prot_local, kind="stable")
    prot_offsets = np.zeros(num_proteins + 1, dtype=np.int64)
    np.cumsum(np.bincount(prot_local, minlength=num_proteins), out=prot_offsets[1:])
    prot_precursors = prec_local[order]

    # precursor -> proteins CSR
    order2 = np.argsort(prec_local, kind="stable")
    prec_offsets = np.zeros(num_precursors + 1, dtype=np.int64)
    np.cumsum(np.bincount(prec_local, minlength=num_precursors), out=prec_offsets[1:])
    prec_proteins = prot_local[order2]

    return (
        protein_ids,
        precursor_ids,
        prot_offsets,
        prot_precursors,
        prec_offsets,
        prec_proteins,
        num_proteins,
        num_precursors,
    )


@nb.njit(cache=True)
def _greedy_parsimony(
    prot_offsets: np.ndarray,
    prot_precursors: np.ndarray,
    prec_offsets: np.ndarray,
    prec_proteins: np.ndarray,
    num_proteins: int,
    num_precursors: int,
) -> np.ndarray:
    """Greedy set cover over CSR adjacency lists.

    Complexity is ``O(P^2 + E)``: the argmax scan is ``O(P)`` per iteration
    (``O(P^2)`` total), while every edge is touched at most once when its
    precursor first becomes covered (``O(E)`` total). Ties are resolved towards
    the smallest local protein index (``>`` is strict), matching ``_build_csr``'s
    ascending ordering.
    """
    protein_lengths = np.zeros(num_proteins, dtype=np.int64)
    for p in range(num_proteins):
        protein_lengths[p] = prot_offsets[p + 1] - prot_offsets[p]

    covered = np.zeros(num_precursors, dtype=np.bool_)
    selected = np.empty(num_proteins, dtype=np.int64)
    n_selected = 0

    for _ in range(num_proteins):
        max_len = 0
        best_protein = -1
        for p in range(num_proteins):
            if protein_lengths[p] > max_len:
                max_len = protein_lengths[p]
                best_protein = p

        if best_protein == -1:
            break

        selected[n_selected] = best_protein
        n_selected += 1

        for e in range(prot_offsets[best_protein], prot_offsets[best_protein + 1]):
            precursor = prot_precursors[e]
            if not covered[precursor]:
                covered[precursor] = True
                for f in range(prec_offsets[precursor], prec_offsets[precursor + 1]):
                    protein_lengths[prec_proteins[f]] -= 1

    return selected[:n_selected]


@nb.njit(cache=True)
def _assemble_output_expanded(
    selected: np.ndarray,
    prot_offsets: np.ndarray,
    prot_precursors: np.ndarray,
    prec_offsets: np.ndarray,
    prec_proteins: np.ndarray,
    num_proteins: int,
    num_precursors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand the selected parsimony proteins back to (precursor, group, master)
    rows, all in local-index space.

    ``master`` is the first protein in greedy-selection order that covers the
    precursor; ``group`` is every selected protein covering it.
    """
    is_selected = np.zeros(num_proteins, dtype=np.bool_)
    master = np.full(num_precursors, -1, dtype=np.int64)

    for s in range(selected.shape[0]):
        p = selected[s]
        is_selected[p] = True
        for e in range(prot_offsets[p], prot_offsets[p + 1]):
            precursor = prot_precursors[e]
            if master[precursor] == -1:
                master[precursor] = p

    total = 0
    for precursor in range(num_precursors):
        if master[precursor] == -1:
            continue
        for f in range(prec_offsets[precursor], prec_offsets[precursor + 1]):
            if is_selected[prec_proteins[f]]:
                total += 1

    out_precursor = np.empty(total, dtype=np.int64)
    out_group = np.empty(total, dtype=np.int64)
    out_master = np.empty(total, dtype=np.int64)

    k = 0
    for precursor in range(num_precursors):
        m = master[precursor]
        if m == -1:
            continue
        for f in range(prec_offsets[precursor], prec_offsets[precursor + 1]):
            p = prec_proteins[f]
            if is_selected[p]:
                out_precursor[k] = precursor
                out_group[k] = p
                out_master[k] = m
                k += 1

    return out_precursor, out_group, out_master


def _protein_group_mapping(
    precursor_index_arr: np.ndarray, protein_index_arr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    (
        protein_ids,
        precursor_ids,
        prot_offsets,
        prot_precursors,
        prec_offsets,
        prec_proteins,
        num_proteins,
        num_precursors,
    ) = _build_csr(precursor_index_arr, protein_index_arr)

    selected = _greedy_parsimony(
        prot_offsets,
        prot_precursors,
        prec_offsets,
        prec_proteins,
        num_proteins,
        num_precursors,
    )

    out_precursor, out_group, out_master = _assemble_output_expanded(
        selected,
        prot_offsets,
        prot_precursors,
        prec_offsets,
        prec_proteins,
        num_proteins,
        num_precursors,
    )

    # Map dense local indices back to original ids.
    return (
        precursor_ids[out_precursor],
        protein_ids[out_group],
        protein_ids[out_master],
    )


def protein_group_mapping(precursor_match_df: pl.DataFrame, fasta_id_df: pl.DataFrame):

    mapping_results = defaultdict(list)

    for is_decoy, sub_df in precursor_match_df.select(
        pl.col("is_decoy", "precursor_index", "protein_index")
    ).group_by("is_decoy"):
        pre_pro_pair_df = sub_df.explode(["protein_index"]).unique(
            subset=["precursor_index", "protein_index"]
        )
        precursor_index_arr, pg_arr, pg_master_arr = _protein_group_mapping(
            pre_pro_pair_df["precursor_index"].to_numpy(),
            pre_pro_pair_df["protein_index"].to_numpy(),
        )
        mapping_results["precursor_index"].append(precursor_index_arr)
        mapping_results["protein_group_index"].append(pg_arr)
        mapping_results["master_protein_index"].append(pg_master_arr)

    mapping_result_df = pl.from_dict(
        {k: np.concatenate(v) for k, v in mapping_results.items()}
    )

    mapping_result_df = (
        mapping_result_df.join(
            fasta_id_df,
            left_on="protein_group_index",
            right_on="protein_index",
            how="left",
        )
        .rename({"fasta_id": "protein_group"})
        .group_by("precursor_index")
        .agg(pl.col("protein_group"), pl.col("master_protein_index").first())
        .join(
            fasta_id_df,
            left_on="master_protein_index",
            right_on="protein_index",
            how="left",
        )
        .rename({"fasta_id": "master_protein"})
        .with_columns(
            pl.col("protein_group").list.sort().list.join(";").alias("protein_group")
        )
    )

    # return precursor_match_df.select(
    #     pl.exclude("protein_group", "master_protein")
    # ).join(
    #     mapping_result_df.select(
    #         pl.col("precursor_index", "protein_group", "master_protein")
    #     ),
    #     on="precursor_index",
    #     how="left",
    # )
    return mapping_result_df.select(
        pl.col("precursor_index", "protein_group", "master_protein")
    )
