from collections import defaultdict
from typing import Tuple

import numpy as np
import numba as nb
import polars as pl


def _build_mappings(precursor_index_arr: np.ndarray, protein_index_arr: np.ndarray):
    protein_to_precursors = defaultdict(set)
    precursor_to_proteins = defaultdict(set)

    for precursor, protein in zip(precursor_index_arr, protein_index_arr):
        protein_to_precursors[protein].add(precursor)
        precursor_to_proteins[precursor].add(protein)

    protein_ids = sorted(protein_to_precursors.keys())
    protein_index_map = {protein: i for i, protein in enumerate(protein_ids)}
    precursor_ids = sorted(precursor_to_proteins.keys())

    pairs = []
    for protein, precursors in protein_to_precursors.items():
        for precursor in precursors:
            pairs.append((protein_index_map[protein], precursor))

    pair_array = np.array(pairs, dtype=np.uint32)

    return pair_array, protein_ids, precursor_ids, protein_to_precursors


@nb.njit(cache=True)
def _greedy_parsimony(pair_array: np.ndarray, num_proteins: int, max_precursor_id: int):

    protein_lengths = np.zeros(num_proteins, dtype=np.uint32)
    covered = np.zeros(max_precursor_id + 1, dtype=np.bool_)
    selected_proteins = []

    for i in range(pair_array.shape[0]):
        protein_idx, precursor_idx = pair_array[i]
        protein_lengths[protein_idx] += 1

    for _ in range(num_proteins):
        max_len = -1
        best_protein = -1
        for i in range(num_proteins):
            if protein_lengths[i] > max_len:
                max_len = protein_lengths[i]
                best_protein = i

        if max_len == 0:
            break

        selected_proteins.append(best_protein)

        for i in range(pair_array.shape[0]):
            protein_idx, precursor_idx = pair_array[i]
            if protein_idx == best_protein and not covered[precursor_idx]:
                covered[precursor_idx] = True
                for j in range(pair_array.shape[0]):
                    if pair_array[j][1] == precursor_idx:
                        protein_lengths[pair_array[j][0]] -= 1

    return np.array(selected_proteins, dtype=np.uint32)


def _assemble_output_expanded(
    precursor_ids: list,
    protein_ids: list,
    selected_protein_indices: np.ndarray,
    protein_to_precursors: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_proteins = [protein_ids[i] for i in selected_protein_indices]

    precursor_to_group = defaultdict(set)
    precursor_counts = defaultdict(int)

    for protein in selected_proteins:
        for precursor in protein_to_precursors[protein]:
            precursor_to_group[precursor].add(protein)
            precursor_counts[(precursor, protein)] += 1

    precursor_list = []
    group_list = []
    master_list = []

    for precursor in precursor_ids:
        group = list(precursor_to_group[precursor])
        master = max(group, key=lambda p: precursor_counts[(precursor, p)])
        for protein in group:
            precursor_list.append(precursor)
            group_list.append(protein)
            master_list.append(master)

    return (
        np.array(precursor_list, dtype=np.uint32),
        np.array(group_list, dtype=np.uint32),
        np.array(master_list, dtype=np.uint32),
    )


def _protein_group_mapping(
    precursor_index_arr: np.ndarray, protein_index_arr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_array, protein_ids, precursor_ids, protein_to_precursors = _build_mappings(
        precursor_index_arr, protein_index_arr
    )
    selected_protein_indices = _greedy_parsimony(
        pair_array, len(protein_ids), max(precursor_index_arr)
    )
    return _assemble_output_expanded(
        precursor_ids, protein_ids, selected_protein_indices, protein_to_precursors
    )


def protein_group_mapping(precursor_match_df: pl.DataFrame, fasta_id_df: pl.DataFrame):

    mapping_results = defaultdict(list)

    for is_decoy, sub_df in precursor_match_df.select(
        pl.col("is_decoy", "precursor_index", "protein_index")
    ).group_by("is_decoy"):
        pre_pro_pair_df = sub_df.explode(["protein_index"])
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
