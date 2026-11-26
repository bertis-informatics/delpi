"""Peptide-level greedy parsimony protein grouping.

Evidence unit is ``peptide_index`` (stripped sequence, shared across charge
states and modifications).

1. Proteins with the same observed peptide set form an equivalence class; one
   representative per class enters greedy selection.
2. Greedy set cover (Numba) over representatives. Ties are broken by a BLAKE2b
   digest of the FASTA accession, so results do not depend on FASTA order,
   protein_index or Python's randomised hash().
3. Each selected class becomes one group. A shared peptide maps to every group
   containing it; groups are never merged.

``lead_only`` groups hold only the representative; ``parsimonious_grouping``
(default) groups hold the whole equivalence class.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Literal, Optional, Tuple

import numba as nb
import numpy as np
import polars as pl


@dataclass
class ProteinGroup:
    """A single inferred protein group.

    Attributes
    ----------
    group_id:
        Group id, stable within one inference run.
    lead_protein_index:
        Original ``protein_index`` of the parsimony lead.
    member_protein_indices:
        Sorted ``protein_index`` values; ``[lead_protein_index]`` for
        ``lead_only``, the whole equivalence class otherwise.
    peptide_indices:
        Sorted unique ``peptide_index`` values of this group.
    """

    group_id: int
    lead_protein_index: int
    member_protein_indices: List[int]
    peptide_indices: List[int]


@dataclass
class ProteinGroupingResult:
    """Output of protein grouping.

    Attributes
    ----------
    groups:
        One row per group: ``group_id``, ``master_protein_index``,
        ``master_protein``, ``protein_group_indices`` (list[int]),
        ``protein_group`` (str), ``peptide_indices`` (list[int]).
    peptide_to_group:
        One row per (peptide_index, group_id) edge: ``peptide_index``,
        ``group_id``, ``master_protein_index``.
    """

    groups: pl.DataFrame
    peptide_to_group: pl.DataFrame


# step 1: CSR graph


def _build_csr(
    peptide_index_arr: np.ndarray,
    protein_index_arr: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int
]:
    """Build CSR adjacency for the protein-peptide graph on dense local indices.

    Returns
    -------
    protein_ids, peptide_ids
        Sorted unique original ids; local index -> original id.
    prot_offsets, prot_peptides
        CSR for protein → peptides.
    pep_offsets, pep_proteins
        CSR for peptide → proteins.
    num_proteins, num_peptides
    """
    protein_ids, prot_local = np.unique(protein_index_arr, return_inverse=True)
    peptide_ids, pep_local = np.unique(peptide_index_arr, return_inverse=True)
    prot_local = prot_local.ravel().astype(np.int64)
    pep_local = pep_local.ravel().astype(np.int64)

    num_proteins = len(protein_ids)
    num_peptides = len(peptide_ids)

    # protein -> peptides CSR
    order = np.argsort(prot_local, kind="stable")
    prot_offsets = np.zeros(num_proteins + 1, dtype=np.int64)
    np.cumsum(np.bincount(prot_local, minlength=num_proteins), out=prot_offsets[1:])
    prot_peptides = pep_local[order]

    # peptide -> proteins CSR
    order2 = np.argsort(pep_local, kind="stable")
    pep_offsets = np.zeros(num_peptides + 1, dtype=np.int64)
    np.cumsum(np.bincount(pep_local, minlength=num_peptides), out=pep_offsets[1:])
    pep_proteins = prot_local[order2]

    return (
        protein_ids,
        peptide_ids,
        prot_offsets,
        prot_peptides,
        pep_offsets,
        pep_proteins,
        num_proteins,
        num_peptides,
    )


# step 2: equivalence classes


def _build_equivalence_classes(
    prot_offsets: np.ndarray,
    prot_peptides: np.ndarray,
    num_proteins: int,
) -> Tuple[Dict[int, FrozenSet[int]], Dict[int, int], np.ndarray]:
    """Group proteins by their exact observed peptide-index set.

    Returns
    -------
    repr_to_members
        Representative local index -> all local indices with the same peptide
        set (including itself).
    protein_to_repr
        Local protein index -> its representative.
    repr_arr
        Sorted representative local indices.
    """
    sig_to_repr: Dict[FrozenSet[int], int] = {}
    repr_to_members: Dict[int, FrozenSet[int]] = {}
    protein_to_repr: Dict[int, int] = {}

    for p in range(num_proteins):
        pep_set = frozenset(
            int(x) for x in prot_peptides[prot_offsets[p] : prot_offsets[p + 1]]
        )
        if pep_set not in sig_to_repr:
            sig_to_repr[pep_set] = p
            repr_to_members[p] = frozenset({p})
            protein_to_repr[p] = p
        else:
            r = sig_to_repr[pep_set]
            repr_to_members[r] = repr_to_members[r] | frozenset({p})
            protein_to_repr[p] = r

    repr_arr = np.array(sorted(repr_to_members.keys()), dtype=np.int64)
    return repr_to_members, protein_to_repr, repr_arr


# step 3: deterministic tie-break ranks


def _blake2b_rank(fasta_ids: np.ndarray) -> np.ndarray:
    """Rank FASTA accessions by BLAKE2b digest (then accession bytes).

    Independent of FASTA order, ``protein_index``, target/entrapment labels and
    Python's ``hash()``. Lower rank wins ties.
    """
    n = len(fasta_ids)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    keys = [
        hashlib.blake2b(str(fid).encode(), digest_size=8).digest() + str(fid).encode()
        for fid in fasta_ids
    ]
    order = sorted(range(n), key=lambda i: keys[i])
    rank = np.empty(n, dtype=np.int64)
    for out_rank, orig_idx in enumerate(order):
        rank[orig_idx] = out_rank
    return rank


# step 4: greedy set cover on representatives


@nb.njit(cache=True)
def _greedy_parsimony(
    repr_indices: np.ndarray,
    prot_offsets: np.ndarray,
    prot_peptides: np.ndarray,
    pep_offsets: np.ndarray,
    pep_proteins: np.ndarray,
    repr_to_repr_local: np.ndarray,
    uncovered_counts: np.ndarray,
    tie_break_rank: np.ndarray,
    priority_score: np.ndarray,
    num_peptides: int,
) -> np.ndarray:
    """Greedy set cover over equivalence-class representatives.

    Each step picks the max uncovered peptide count, then max
    ``priority_score`` (0 if unused), then min ``tie_break_rank``.

    Parameters
    ----------
    repr_indices:
        Local protein indices of the equivalence-class representatives.
    prot_offsets, prot_peptides, pep_offsets, pep_proteins:
        CSR adjacency over all proteins.
    repr_to_repr_local:
        Local protein index -> position in ``repr_indices``, -1 if not a
        representative.
    uncovered_counts:
        Uncovered-peptide count per representative; modified in place.
    tie_break_rank, priority_score:
        Aligned with ``repr_indices``.

    Returns
    -------
    selected_repr_positions : np.ndarray
        Positions in ``repr_indices``, in selection order.
    """
    n_repr = repr_indices.shape[0]
    covered = np.zeros(num_peptides, dtype=nb.boolean)
    selected = np.empty(n_repr, dtype=np.int64)
    n_selected = 0

    for _ in range(n_repr):
        best_pos = -1
        best_count = np.int64(0)
        best_prio = priority_score[0] - priority_score[0]  # typed zero
        best_rank = np.int64(9223372036854775807)  # int64 max

        for i in range(n_repr):
            c = uncovered_counts[i]
            if c <= 0:
                continue
            prio = priority_score[i]
            rank = tie_break_rank[i]
            if (
                c > best_count
                or (c == best_count and prio > best_prio)
                or (c == best_count and prio == best_prio and rank < best_rank)
            ):
                best_count = c
                best_prio = prio
                best_rank = rank
                best_pos = i

        if best_pos == -1:
            break

        selected[n_selected] = best_pos
        n_selected += 1

        # cover the lead's peptides and decrement counts of affected reprs
        p_global = repr_indices[best_pos]
        for e in range(prot_offsets[p_global], prot_offsets[p_global + 1]):
            pep = prot_peptides[e]
            if not covered[pep]:
                covered[pep] = True
                for f in range(pep_offsets[pep], pep_offsets[pep + 1]):
                    nb_prot = pep_proteins[f]
                    r_pos = repr_to_repr_local[nb_prot]
                    if r_pos >= 0:
                        uncovered_counts[r_pos] -= 1

    return selected[:n_selected]


# step 5: assemble output


def _assemble_groups(
    selected_repr_positions: np.ndarray,
    repr_indices: np.ndarray,
    repr_to_members: Dict[int, FrozenSet[int]],
    prot_offsets: np.ndarray,
    prot_peptides: np.ndarray,
    protein_ids: np.ndarray,
    peptide_ids: np.ndarray,
    fasta_id_map: Dict[int, str],
    grouping_type: Literal["lead_only", "parsimonious_grouping"],
) -> "ProteinGroupingResult":
    """Build a ProteinGroupingResult, one group per selected representative.

    A peptide shared by several groups gets one ``peptide_to_group`` row per
    group.

    Parameters
    ----------
    repr_to_members:
        Representative local index -> member local indices (used only for
        ``parsimonious_grouping``).
    protein_ids, peptide_ids:
        Local-to-original id lookup.
    fasta_id_map:
        Original protein_index -> FASTA accession.
    """
    groups_rows: List[dict] = []
    pep2grp_peptide: List[int] = []
    pep2grp_group: List[int] = []
    pep2grp_master: List[int] = []

    seen_pep_group: set = set()

    for group_id, pos in enumerate(selected_repr_positions):
        lead_local = int(repr_indices[pos])
        lead_orig = int(protein_ids[lead_local])

        if grouping_type == "parsimonious_grouping":
            member_locals = sorted(repr_to_members[lead_local])
        else:
            member_locals = [lead_local]

        member_orig = sorted(int(protein_ids[m]) for m in member_locals)

        # Observed peptide set for the lead (identical for all class members)
        pep_orig = sorted(
            int(peptide_ids[int(prot_peptides[e])])
            for e in range(prot_offsets[lead_local], prot_offsets[lead_local + 1])
        )

        member_accessions = sorted(
            fasta_id_map.get(orig, str(orig)) for orig in member_orig
        )
        lead_accession = fasta_id_map.get(lead_orig, str(lead_orig))

        groups_rows.append(
            {
                "group_id": group_id,
                "master_protein_index": lead_orig,
                "master_protein": lead_accession,
                "protein_group_indices": member_orig,
                "protein_group": ";".join(member_accessions),
                "peptide_indices": pep_orig,
            }
        )

        for pep_o in pep_orig:
            key = (pep_o, group_id)
            if key not in seen_pep_group:
                seen_pep_group.add(key)
                pep2grp_peptide.append(pep_o)
                pep2grp_group.append(group_id)
                pep2grp_master.append(lead_orig)

    groups_df = pl.DataFrame(
        groups_rows,
        schema={
            "group_id": pl.Int64,
            "master_protein_index": pl.Int64,
            "master_protein": pl.String,
            "protein_group_indices": pl.List(pl.Int64),
            "protein_group": pl.String,
            "peptide_indices": pl.List(pl.Int64),
        },
    )

    peptide_to_group_df = pl.DataFrame(
        {
            "peptide_index": pl.Series(pep2grp_peptide, dtype=pl.Int64),
            "group_id": pl.Series(pep2grp_group, dtype=pl.Int64),
            "master_protein_index": pl.Series(pep2grp_master, dtype=pl.Int64),
        }
    )

    return ProteinGroupingResult(
        groups=groups_df,
        peptide_to_group=peptide_to_group_df,
    )


# top-level inference


def infer_protein_groups(
    peptide_index_arr: np.ndarray,
    protein_index_arr: np.ndarray,
    fasta_ids: np.ndarray,
    grouping_type: Literal[
        "lead_only", "parsimonious_grouping"
    ] = "parsimonious_grouping",
    priority_score: Optional[np.ndarray] = None,
) -> "ProteinGroupingResult":
    """Peptide-level greedy parsimony protein inference.

    Parameters
    ----------
    peptide_index_arr, protein_index_arr:
        Parallel arrays of observed (peptide_index, protein_index) edges.
    fasta_ids:
        FASTA accessions indexed by original ``protein_index``; must cover
        every protein in ``protein_index_arr``.
    priority_score:
        Optional, indexed by original ``protein_index``; higher wins greedy
        ties. ``None`` means all 0.
    """
    if len(peptide_index_arr) == 0:
        empty_groups = pl.DataFrame(
            schema={
                "group_id": pl.Int64,
                "master_protein_index": pl.Int64,
                "master_protein": pl.String,
                "protein_group_indices": pl.List(pl.Int64),
                "protein_group": pl.String,
                "peptide_indices": pl.List(pl.Int64),
            }
        )
        empty_p2g = pl.DataFrame(
            schema={
                "peptide_index": pl.Int64,
                "group_id": pl.Int64,
                "master_protein_index": pl.Int64,
            }
        )
        return ProteinGroupingResult(groups=empty_groups, peptide_to_group=empty_p2g)

    (
        protein_ids,
        peptide_ids,
        prot_offsets,
        prot_peptides,
        pep_offsets,
        pep_proteins,
        num_proteins,
        num_peptides,
    ) = _build_csr(peptide_index_arr, protein_index_arr)

    # initial representative = lowest local index
    repr_to_members, _protein_to_repr, repr_arr = _build_equivalence_classes(
        prot_offsets, prot_peptides, num_proteins
    )
    n_repr = len(repr_arr)

    fasta_id_map: Dict[int, str] = {
        int(protein_ids[local]): str(fasta_ids[protein_ids[local]])
        for local in range(num_proteins)
    }

    # re-elect each class's representative by BLAKE2b rank so it does not
    # depend on protein_index / FASTA order
    new_repr_to_members: Dict[int, FrozenSet[int]] = {}
    for old_repr, members in repr_to_members.items():
        member_locals = sorted(members)
        member_orig_ids = [int(protein_ids[m]) for m in member_locals]
        member_fasta = np.array(
            [fasta_id_map.get(oid, str(oid)) for oid in member_orig_ids], dtype=object
        )
        ranks = _blake2b_rank(member_fasta)
        best_pos = int(np.argmin(ranks))
        new_repr = member_locals[best_pos]
        new_repr_to_members[new_repr] = frozenset(members)

    repr_to_members = new_repr_to_members
    repr_arr = np.array(sorted(repr_to_members.keys()), dtype=np.int64)
    n_repr = len(repr_arr)

    repr_orig_ids = np.array(
        [int(protein_ids[int(r)]) for r in repr_arr], dtype=np.int64
    )
    repr_fasta_ids = np.array(
        [fasta_id_map.get(int(oid), str(oid)) for oid in repr_orig_ids],
        dtype=object,
    )
    tie_break_rank = _blake2b_rank(repr_fasta_ids)

    if priority_score is not None:
        prio_arr = np.array(
            [float(priority_score[int(protein_ids[int(r)])]) for r in repr_arr],
            dtype=np.float64,
        )
    else:
        prio_arr = np.zeros(n_repr, dtype=np.float64)

    uncovered_counts = np.array(
        [int(prot_offsets[int(r) + 1] - prot_offsets[int(r)]) for r in repr_arr],
        dtype=np.int64,
    )

    # local protein index -> position in repr_arr, -1 if not a representative
    repr_to_repr_local = np.full(num_proteins, -1, dtype=np.int64)
    for pos, r in enumerate(repr_arr):
        repr_to_repr_local[int(r)] = pos

    selected_positions = _greedy_parsimony(
        repr_arr,
        prot_offsets,
        prot_peptides,
        pep_offsets,
        pep_proteins,
        repr_to_repr_local,
        uncovered_counts,
        tie_break_rank,
        prio_arr,
        num_peptides,
    )

    return _assemble_groups(
        selected_positions,
        repr_arr,
        repr_to_members,
        prot_offsets,
        prot_peptides,
        protein_ids,
        peptide_ids,
        fasta_id_map,
        grouping_type,
    )


# public wrapper


def protein_group_mapping(
    pmsm_df: pl.DataFrame,
    fasta_id_df: pl.DataFrame,
    grouping_type: Literal[
        "lead_only", "parsimonious_grouping"
    ] = "parsimonious_grouping",
) -> pl.DataFrame:
    """Map peptides to protein groups, targets and decoys separately.

    Returns one row per (peptide_index, group_id) edge, so shared peptides
    appear in every group that contains them.

    Parameters
    ----------
    pmsm_df:
        Must contain columns: ``is_decoy`` (bool), ``peptide_index`` (int),
        ``protein_index`` (list[int]).
    fasta_id_df:
        Must contain columns: ``protein_index`` (int), ``fasta_id`` (str).
        Each unique ``protein_index`` must appear at most once.
    grouping_type:
        ``"lead_only"`` or ``"parsimonious_grouping"`` (default).

    Returns
    -------
    pl.DataFrame
        Columns: ``peptide_index``, ``group_id``, ``protein_group`` (str),
        ``master_protein`` (str), ``master_protein_index`` (int).
    """
    fasta_lookup: Dict[int, str] = dict(
        zip(
            fasta_id_df["protein_index"].to_list(),
            fasta_id_df["fasta_id"].to_list(),
        )
    )
    if len(fasta_id_df) == 0:
        fasta_arr = np.array([], dtype=object)
    else:
        max_prot_idx = int(fasta_id_df["protein_index"].max())
        fasta_arr = np.empty(max_prot_idx + 1, dtype=object)
        for pid, fid in fasta_lookup.items():
            fasta_arr[pid] = fid

    parts: List[pl.DataFrame] = []
    group_id_offset = 0

    for (is_decoy,), sub_df in pmsm_df.select(
        pl.col("is_decoy", "peptide_index", "protein_index")
    ).group_by("is_decoy"):
        pair_df = sub_df.explode("protein_index", empty_as_null=True).unique(
            subset=["peptide_index", "protein_index"]
        )

        peptide_index_arr = pair_df["peptide_index"].to_numpy()
        protein_index_arr = pair_df["protein_index"].to_numpy()

        result = infer_protein_groups(
            peptide_index_arr,
            protein_index_arr,
            fasta_arr,
            grouping_type=grouping_type,
        )

        if result.peptide_to_group.is_empty():
            continue

        # keep group_ids unique across target/decoy
        p2g = result.peptide_to_group.with_columns(
            (pl.col("group_id") + group_id_offset).alias("group_id")
        )
        grp = result.groups.with_columns(
            (pl.col("group_id") + group_id_offset).alias("group_id")
        )
        group_id_offset += result.groups.height

        p2g = p2g.join(
            grp.select("group_id", "protein_group", "master_protein"),
            on="group_id",
            how="left",
        )
        parts.append(p2g)

    if not parts:
        return pl.DataFrame(
            schema={
                "peptide_index": pl.Int64,
                "group_id": pl.Int64,
                "master_protein_index": pl.Int64,
                "protein_group": pl.String,
                "master_protein": pl.String,
            }
        )

    return pl.concat(parts, how="vertical").select(
        "peptide_index",
        "group_id",
        "master_protein_index",
        "protein_group",
        "master_protein",
    )
