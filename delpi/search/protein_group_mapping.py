"""Peptide-level greedy parsimony protein grouping.

Design overview
---------------
Evidence unit
    ``peptide_index`` (stripped peptide sequence identifier, shared across
    charge states / modifications).

Pipeline
    1. Build equivalence classes of proteins that share the *exact same*
       observed peptide set.  One representative per class enters greedy
       selection; all class members are tracked alongside it.
    2. Run greedy set-cover (Numba) on the equivalence-class representatives.
       Ties are broken deterministically by a BLAKE2b digest of the FASTA
       accession – independent of FASTA row order, protein_index, or Python's
       randomised hash().
    3. Assemble *group-centric* output: each selected equivalence class becomes
       exactly one ``ProteinGroupingResult`` group.  A shared peptide maps to
       *all* groups that contain it (many-to-many), never causing independent
       groups to be merged.

Grouping types
    ``lead_only``
        Each group contains only the selected representative protein.
    ``parsimonious_grouping`` (default)
        Each group contains all proteins whose observed peptide set is
        *exactly identical* to the selected representative's set (i.e. all
        members of the same equivalence class).
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Literal, Optional, Tuple

import numba as nb
import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Public result containers
# ---------------------------------------------------------------------------


@dataclass
class ProteinGroup:
    """A single inferred protein group.

    Attributes
    ----------
    group_id:
        Stable integer identifier for this group within one inference run.
    lead_protein_index:
        Original ``protein_index`` of the selected parsimony lead.
    member_protein_indices:
        Sorted list of all ``protein_index`` values belonging to this group.
        For ``lead_only`` this is always ``[lead_protein_index]``.
        For ``parsimonious_grouping`` this includes every protein whose
        observed peptide set equals the lead's set exactly.
    peptide_indices:
        Sorted unique list of ``peptide_index`` values observed for this group.
    """

    group_id: int
    lead_protein_index: int
    member_protein_indices: List[int]
    peptide_indices: List[int]


@dataclass
class ProteinGroupingResult:
    """Full output of the protein-grouping algorithm.

    Attributes
    ----------
    groups:
        One row per inferred protein group.
        Columns: ``group_id``, ``master_protein_index``, ``master_protein``,
        ``protein_group_indices`` (list[int]), ``protein_group`` (str),
        ``peptide_indices`` (list[int]).
    peptide_to_group:
        One row per **(peptide_index, group_id)** edge.  A shared peptide
        appears once per group that covers it.
        Columns: ``peptide_index``, ``group_id``, ``master_protein_index``.
    """

    groups: pl.DataFrame
    peptide_to_group: pl.DataFrame


# ---------------------------------------------------------------------------
# Step 1 – CSR graph construction
# ---------------------------------------------------------------------------


def _build_csr(
    peptide_index_arr: np.ndarray,
    protein_index_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Build CSR adjacency lists for the bipartite protein <-> peptide graph.

    All protein and peptide identifiers are remapped to dense local indices
    (0..P-1, 0..N-1) so that the Numba kernel sees compact integer arrays.
    The mapping back to original ids is provided by the returned
    ``protein_ids`` / ``peptide_ids`` arrays.

    Returns
    -------
    protein_ids, peptide_ids
        Sorted unique original ids; local_index → original_id.
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


# ---------------------------------------------------------------------------
# Step 2 – Equivalence classes
# ---------------------------------------------------------------------------


def _build_equivalence_classes(
    prot_offsets: np.ndarray,
    prot_peptides: np.ndarray,
    num_proteins: int,
) -> Tuple[Dict[int, FrozenSet[int]], Dict[int, int], np.ndarray]:
    """Group proteins by their exact observed peptide-index set.

    Returns
    -------
    repr_to_members
        Mapping from representative local protein index → frozenset of all
        local protein indices with the same peptide set (including the
        representative itself).
    protein_to_repr
        Mapping from every local protein index → its representative.
    repr_arr
        Sorted array of representative local protein indices (the nodes that
        enter greedy selection).
    """
    sig_to_repr: Dict[FrozenSet[int], int] = {}
    repr_to_members: Dict[int, FrozenSet[int]] = {}
    protein_to_repr: Dict[int, int] = {}

    for p in range(num_proteins):
        pep_set = frozenset(int(x) for x in prot_peptides[prot_offsets[p]: prot_offsets[p + 1]])
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


# ---------------------------------------------------------------------------
# Step 3 – Deterministic tie-break ranks
# ---------------------------------------------------------------------------


def _blake2b_rank(fasta_ids: np.ndarray) -> np.ndarray:
    """Compute a stable integer rank for each FASTA accession.

    The rank is derived from the BLAKE2b digest of each accession string so
    that it is independent of FASTA row order, ``protein_index``, target/
    entrapment labels, and Python's randomised built-in ``hash()``.

    Lower rank = preferred in tie-breaking (wins greedy selection when
    uncovered-peptide counts are equal).

    Implementation note: sorting is done on (digest_bytes, accession_bytes)
    tuples using Python's built-in ``sorted()`` with a key function to avoid
    NumPy's dtype restrictions on object arrays.
    """
    n = len(fasta_ids)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    keys = [
        hashlib.blake2b(str(fid).encode(), digest_size=8).digest() + str(fid).encode()
        for fid in fasta_ids
    ]
    # Use Python sorted (stable, works on bytes) to get the sorted order
    order = sorted(range(n), key=lambda i: keys[i])
    rank = np.empty(n, dtype=np.int64)
    for out_rank, orig_idx in enumerate(order):
        rank[orig_idx] = out_rank
    return rank


# ---------------------------------------------------------------------------
# Step 4 – Numba greedy set cover on equivalence-class representatives
# ---------------------------------------------------------------------------


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
    """Greedy set cover over the equivalence-class representative proteins.

    Selection criterion (lexicographic, lower is better for rank):
        1. Maximize currently uncovered peptide count.
        2. Maximize ``priority_score`` (e.g. protein evidence; 0 if unused).
        3. Minimize ``tie_break_rank`` (BLAKE2b-derived; independent of DB order).

    Parameters
    ----------
    repr_indices:
        Local protein indices of the equivalence-class representatives.
    prot_offsets, prot_peptides:
        CSR: protein → peptides (over the *full* protein set).
    pep_offsets, pep_proteins:
        CSR: peptide → proteins (over the *full* protein set).
    repr_to_repr_local:
        Maps each global local protein index to its position in ``repr_indices``
        (or -1 if it is not a representative).
    uncovered_counts:
        Mutable working copy of the uncovered-peptide count per representative.
    tie_break_rank, priority_score:
        Per-representative arrays aligned with ``repr_indices``.
    num_peptides:
        Total number of distinct local peptide indices.

    Returns
    -------
    selected_repr_positions : np.ndarray
        Positions in ``repr_indices`` of the selected representatives, in
        greedy-selection order.
    """
    n_repr = repr_indices.shape[0]
    covered = np.zeros(num_peptides, dtype=nb.boolean)
    selected = np.empty(n_repr, dtype=np.int64)
    n_selected = 0

    for _ in range(n_repr):
        # Lexicographic argmax: (uncovered_count, priority_score, -tie_break_rank)
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
            if (c > best_count
                    or (c == best_count and prio > best_prio)
                    or (c == best_count and prio == best_prio and rank < best_rank)):
                best_count = c
                best_prio = prio
                best_rank = rank
                best_pos = i

        if best_pos == -1:
            break

        selected[n_selected] = best_pos
        n_selected += 1

        # Mark all newly covered peptides and update counts of affected reprs
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


# ---------------------------------------------------------------------------
# Step 5 – Assemble group-centric output
# ---------------------------------------------------------------------------


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
    """Build a ProteinGroupingResult from the greedy-selected representatives.

    Each selected equivalence-class representative produces *exactly one*
    protein group.  The many-to-many peptide→group mapping is preserved: a
    peptide shared by two independent groups appears twice in
    ``peptide_to_group``, once per group.

    Parameters
    ----------
    selected_repr_positions:
        Positions in ``repr_indices`` chosen by greedy selection.
    repr_indices:
        Local protein indices of all equivalence-class representatives.
    repr_to_members:
        Mapping from representative local index → frozenset of all member
        local protein indices (used only for ``parsimonious_grouping``).
    prot_offsets, prot_peptides:
        CSR protein → peptides (local index space).
    protein_ids, peptide_ids:
        Dense-to-original id lookup arrays.
    fasta_id_map:
        Mapping from original protein_index → FASTA accession string.
    grouping_type:
        ``"lead_only"``  – group members = {lead only}.
        ``"parsimonious_grouping"`` – group members = entire equivalence class.
    """
    groups_rows: List[dict] = []
    pep2grp_peptide: List[int] = []
    pep2grp_group: List[int] = []
    pep2grp_master: List[int] = []

    seen_pep_group: set = set()

    for group_id, pos in enumerate(selected_repr_positions):
        lead_local = int(repr_indices[pos])
        lead_orig = int(protein_ids[lead_local])

        # Determine group member original protein indices
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

        # FASTA accessions
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

        # Many-to-many peptide → group edges (deduplicated)
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


# ---------------------------------------------------------------------------
# Top-level inference function (returns ProteinGroupingResult)
# ---------------------------------------------------------------------------


def infer_protein_groups(
    peptide_index_arr: np.ndarray,
    protein_index_arr: np.ndarray,
    fasta_ids: np.ndarray,
    grouping_type: Literal["lead_only", "parsimonious_grouping"] = "parsimonious_grouping",
    priority_score: Optional[np.ndarray] = None,
) -> "ProteinGroupingResult":
    """Run full peptide-level greedy parsimony protein inference.

    Parameters
    ----------
    peptide_index_arr, protein_index_arr:
        Parallel arrays of (peptide_index, protein_index) edges after
        exploding the per-precursor protein list.  One edge = one observed
        peptide for one protein.
    fasta_ids:
        Array of FASTA accession strings, indexed by *original*
        ``protein_index`` (i.e. ``fasta_ids[protein_index]`` → accession).
        Must cover every protein_index present in ``protein_index_arr``.
    grouping_type:
        ``"lead_only"`` or ``"parsimonious_grouping"`` (default).
    priority_score:
        Optional float array indexed by *original* ``protein_index``.
        Higher score = preferred in greedy tie-breaking.  When ``None``,
        all proteins receive score 0.

    Returns
    -------
    ProteinGroupingResult
        See the dataclass docstring for the output schema.
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

    # Build equivalence classes (initial representative = lowest local index)
    repr_to_members, _protein_to_repr, repr_arr = _build_equivalence_classes(
        prot_offsets, prot_peptides, num_proteins
    )
    n_repr = len(repr_arr)

    # Build fasta_id lookup over original protein indices
    fasta_id_map: Dict[int, str] = {
        int(protein_ids[local]): str(fasta_ids[protein_ids[local]])
        for local in range(num_proteins)
    }

    # Re-elect representatives using BLAKE2b rank so that the choice is
    # independent of protein_index / FASTA row order.
    # For each equivalence class, pick the member with the smallest rank.
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

    # Tie-break ranks over original protein ids of representatives
    repr_orig_ids = np.array(
        [int(protein_ids[int(r)]) for r in repr_arr], dtype=np.int64
    )
    repr_fasta_ids = np.array(
        [fasta_id_map.get(int(oid), str(oid)) for oid in repr_orig_ids],
        dtype=object,
    )
    tie_break_rank = _blake2b_rank(repr_fasta_ids)

    # Priority scores aligned to repr_arr
    if priority_score is not None:
        prio_arr = np.array(
            [float(priority_score[int(protein_ids[int(r)])]) for r in repr_arr],
            dtype=np.float64,
        )
    else:
        prio_arr = np.zeros(n_repr, dtype=np.float64)

    # Initial uncovered counts per representative
    uncovered_counts = np.array(
        [
            int(prot_offsets[int(r) + 1] - prot_offsets[int(r)])
            for r in repr_arr
        ],
        dtype=np.int64,
    )

    # repr_to_repr_local: maps each local protein index → position in repr_arr (-1 if not a repr)
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


# ---------------------------------------------------------------------------
# Public compatibility wrapper  (used by FDRAnalyzer)
# ---------------------------------------------------------------------------


def protein_group_mapping(
    pmsm_df: pl.DataFrame,
    fasta_id_df: pl.DataFrame,
    grouping_type: Literal["lead_only", "parsimonious_grouping"] = "parsimonious_grouping",
) -> pl.DataFrame:
    """Map confident peptide identifications to protein groups.

    This is the public API consumed by :class:`FDRAnalyzer`.  It processes
    target and decoy peptides independently (preserving the existing
    target/decoy split behaviour) and returns a flat DataFrame with one row
    per **(peptide_index, group_id)** edge so that shared peptides correctly
    appear in multiple groups.

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

        One row per (peptide_index, group_id) edge.  A peptide shared by two
        independent protein groups produces two rows with distinct ``group_id``
        values and therefore distinct ``protein_group`` values.
    """
    # Build a fast protein_index -> fasta_id lookup
    fasta_lookup: Dict[int, str] = dict(
        zip(
            fasta_id_df["protein_index"].to_list(),
            fasta_id_df["fasta_id"].to_list(),
        )
    )
    # Max protein_index for array indexing
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
        pair_df = sub_df.explode("protein_index").unique(
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

        # Shift group_ids so they don't collide across target/decoy halves
        p2g = result.peptide_to_group.with_columns(
            (pl.col("group_id") + group_id_offset).alias("group_id")
        )
        grp = result.groups.with_columns(
            (pl.col("group_id") + group_id_offset).alias("group_id")
        )
        group_id_offset += result.groups.height

        # Attach protein_group / master_protein strings to the edge table
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
