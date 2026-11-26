import logging
import random
from collections import Counter
from functools import partial
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

import polars as pl

from delpi.chem.amino_acid import AminoAcid
from delpi.chem.modification_param import ModificationLocation
from delpi.database.utils import il_normalized_sequence, il_normalized_sequence_expr

MUTATION_MAP = dict(zip("GAVLIFMPWSCTYHKRQEND", "LLLVVLLLLTSSSSLLNDQE"))

logger = logging.getLogger(__name__)


def _default_varmod_signature(aa: str) -> tuple:
    """Signature without variable modifications: every position is safe."""
    return ()


def build_varmod_signature_fn(
    mod_param_set: Optional[Sequence],
) -> Callable[[str], tuple]:
    """Map a residue to its eligibility tuple over the variable modifications.

    Residues are grouped by modification name, so Phospho(S)/(T)/(Y) give one
    flag, as in ``ModificationHandler.generate_variable_modifications``. A
    mutation is safe when both residues have the same tuple. Fixed modifications
    are ignored.
    """
    mod_name_to_residues: dict = {}
    for p in mod_param_set or []:
        if p.fixed:
            continue
        mod_name_to_residues.setdefault(p.mod_name, set()).add(p.residue)

    if not mod_name_to_residues:
        return _default_varmod_signature

    mod_names = sorted(mod_name_to_residues)

    def signature(aa: str) -> tuple:
        return tuple(aa in mod_name_to_residues[name] for name in mod_names)

    return signature


def _find_safe_mutation_position(
    peptide: str,
    candidates: Sequence[int],
    varmod_signature: Callable[[str], tuple],
    exclude: Optional[int] = None,
) -> Optional[int]:
    """First of ``candidates`` whose ``MUTATION_MAP`` substitution keeps the
    residue's varmod signature, or None.

    Negative indices count from the end. Positions 0 and len - 1 and
    ``exclude`` are skipped.
    """
    length = len(peptide)
    for pos in candidates:
        if pos < 0:
            pos += length
        if pos <= 0 or pos >= length - 1 or pos == exclude:
            continue
        aa = peptide[pos]
        if varmod_signature(aa) == varmod_signature(MUTATION_MAP[aa]):
            return pos
    return None


def get_mutated_decoy(
    peptide: str, varmod_signature: Callable[[str], tuple] = None
) -> str:
    """Decoy with one N-side and one C-side residue replaced via MUTATION_MAP.

    The first character and the last two are kept. Each side mutates the first
    candidate whose substitution keeps the residue's varmod eligibility (see
    ``build_varmod_signature_fn``): N side 2, 3, 4, 1; C side len - 3, len - 4,
    len - 5, skipping the N-side position. A side with no safe candidate mutates
    its default (2 or len - 3) unconditionally, so the decoy always differs from
    the target.
    """
    if varmod_signature is None:
        varmod_signature = _default_varmod_signature

    n_default = 2
    c_default = len(peptide) - 3

    n_pos = _find_safe_mutation_position(peptide, (2, 3, 4, 1), varmod_signature)
    if n_pos is None:
        n_pos = n_default

    c_pos = _find_safe_mutation_position(
        peptide, (-3, -4, -5), varmod_signature, exclude=n_pos
    )
    if c_pos is None:
        c_pos = c_default

    chars = list(peptide)
    chars[n_pos] = MUTATION_MAP[peptide[n_pos]]
    chars[c_pos] = MUTATION_MAP[peptide[c_pos]]

    return "".join(chars)


def get_shuffled_decoy(peptide):
    mid = list(peptide[1:-2])
    random.shuffle(mid)
    return f"{peptide[0]}{''.join(mid)}{peptide[-2:]}"


# Swap decoys (DIA-NN style, composition preserving). Peptide strings carry
# terminal markers at peptide[0] and peptide[-1].
ResidueMasses = Dict[str, Tuple[float, ...]]


def build_residue_masses(mod_param_set: Optional[Sequence] = None) -> ResidueMasses:
    """Possible masses of each standard residue at an internal position.

    Element 0 is the residue mass including fixed 'anywhere' modifications;
    the rest add each variable 'anywhere' modification. Terminal-specific
    modifications are ignored because swap decoys never change the first or
    last residue.
    """
    base = {aa.residue: aa.mass for aa in AminoAcid.standard_amino_acids}
    var_deltas = {r: [] for r in base}
    for p in mod_param_set or []:
        if p.location != ModificationLocation.ANYWHERE or p.residue not in base:
            continue
        if p.fixed:
            base[p.residue] += p.modification.mass
        else:
            var_deltas[p.residue].append(p.modification.mass)
    return {r: (m, *(m + d for d in var_deltas[r])) for r, m in base.items()}


def _robust_shift_count(
    peptide: str, perm: Sequence[int], residue_masses: ResidueMasses, min_shift: float
) -> int:
    """Number of b ions of the decoy ``peptide[perm]`` that move by >= ``min_shift``
    in every modification state of the moved residues.

    A residue keeps its modification state when it moves, so for a single
    transposition this is ``span`` if the pair is in ``DecoyGenerator._swap_pairs``
    and 0 otherwise: swaps and shuffles obey the same rule. ``perm`` must keep
    the terminal markers in place. y ions mirror b ions (same precursor mass).
    """
    count, in_target, in_decoy = 0, set(), set()
    for k in range(len(peptide) - 1):
        in_target.add(k)
        in_decoy.add(perm[k])
        deltas = {0.0}
        for sign, moved in ((1, in_target - in_decoy), (-1, in_decoy - in_target)):
            for i in moved:
                deltas = {
                    round(d + sign * m, 4)
                    for d in deltas
                    for m in residue_masses[peptide[i]]
                }
        count += min(map(abs, deltas)) >= min_shift
    return count


class DecoyGenerator:

    supported_methods = [
        "pseudo_reverse",
        "mutation",
        "pseudo_shuffle",
        "swap",
    ]

    def __init__(
        self,
        method: str = None,
        random_seed: int = 323,
        mod_param_set: Optional[Sequence] = None,
        min_swap_mass_shift: float = 5.0,
        max_shuffle_attempts: int = 20,
    ):

        if method is not None and method not in self.supported_methods:
            raise ValueError(
                f"Unsupported decoy generation method: {method}. "
                f"Supported methods: {self.supported_methods}"
            )
        self.method = method
        self.mod_param_set = mod_param_set
        self._varmod_signature = build_varmod_signature_fn(mod_param_set)
        self.random_seed = random_seed

        # swap decoy settings
        self.min_swap_mass_shift = min_swap_mass_shift
        self.max_shuffle_attempts = max_shuffle_attempts
        self._residue_masses = masses = build_residue_masses(mod_param_set)
        # residue pairs whose exchange moves fragments by >= min_swap_mass_shift
        # in every fixed/variable modification state (excludes I/L, N/D, Q/K/E,
        # F/M(ox), R/C(cam), ...)
        self._swap_pairs = frozenset(
            (a, b)
            for a in masses
            for b in masses
            if a.replace("I", "L") != b.replace("I", "L")
            and min(abs(x - y) for x in masses[a] for y in masses[b])
            >= min_swap_mass_shift
        )

    def _swap_candidates(self, peptide: str):
        """Yield ``(decoy, decoy_type)`` in preference order.

        Terminal markers and the first and last residues never change, and no
        candidate equals the target (I == L) by construction.

        1. ``swap``: exchange residues 2 and n-1 (b2..b_{n-2}, y2..y_{n-2} move).
        2. ``swap_alt``: other internal transpositions, widest span first;
           span = number of moved b ions >= ceil((n-3)/2).
        3. ``shuffle``: seeded shuffles of residues 2..n-1, accepted if
           >= ceil((n-3)/2) b ions move in every modification state.
        4. ``mutation``: one or two internal residues via ``MUTATION_MAP``
           (changes mass); varmod-eligibility-preserving sites first.
        """
        p, L = peptide, len(peptide)  # residues p[1:L-1], n = L - 2
        min_shifted = max(1, (L - 4) // 2)  # ceil((n - 3) / 2)

        pairs = self._swap_pairs
        for span in range(L - 5, min_shifted - 1, -1):
            for i in range(2, L - 2 - span):
                j = i + span
                if (p[i], p[j]) in pairs:
                    decoy_type = "swap" if span == L - 5 else "swap_alt"
                    yield f"{p[:i]}{p[j]}{p[i + 1:j]}{p[i]}{p[j + 1:]}", decoy_type

        interior = list(range(2, L - 2))
        if len(set(p[2 : L - 2].replace("I", "L"))) > 1:
            rng = random.Random(f"{self.random_seed}:{p}")
            for _ in range(self.max_shuffle_attempts):
                rng.shuffle(interior)
                perm = [0, 1, *interior, L - 2, L - 1]
                shifted = _robust_shift_count(
                    p, perm, self._residue_masses, self.min_swap_mass_shift
                )
                if shifted >= min_shifted:
                    yield "".join(p[k] for k in perm), "shuffle"

        sig = self._varmod_signature
        safe = lambda i: sig(p[i]) == sig(MUTATION_MAP[p[i]])
        inner = range(2, L - 2)
        sites = [(i, j) for i in inner for j in inner if i < j] or [(i,) for i in inner]
        sites.sort(key=lambda s: (not all(map(safe, s)), s[0] - s[-1], s[0]))
        for site in sites:
            chars = list(p)
            for i in site:
                chars[i] = MUTATION_MAP[chars[i]]
            yield "".join(chars), "mutation"

    def _swap_decoy(
        self, peptide: str, forbidden: Set[str]
    ) -> Tuple[Optional[str], str]:
        """First candidate whose I/L-normalized sequence is not in ``forbidden``
        (it is then added to it). ``(None, "failed")`` if none, e.g. n <= 2."""
        for decoy, decoy_type in self._swap_candidates(peptide):
            key = il_normalized_sequence(decoy)
            if key not in forbidden:
                forbidden.add(key)
                return decoy, decoy_type
        return None, "failed"

    def _swap_decoys(self, peptides: Sequence[str]) -> Tuple[list, list]:
        """Row-aligned swap decoys (``None`` where failed).

        Targets are processed in sorted peptide order, so the result does not
        depend on row order. Targets and accepted decoys are blocked, so decoys
        are unique and never equal a target (I == L, markers ignored).
        """
        forbidden = {il_normalized_sequence(p) for p in peptides}
        decoys, decoy_types = [None] * len(peptides), ["failed"] * len(peptides)
        for i in sorted(range(len(peptides)), key=peptides.__getitem__):
            decoys[i], decoy_types[i] = self._swap_decoy(peptides[i], forbidden)

        counts = Counter(decoy_types)
        logger.info("Swap decoy types: %s", dict(counts))
        if counts["failed"]:
            logger.warning(
                "%d target peptides have no swap decoy (no accepted candidate)",
                counts["failed"],
            )
        return decoys, decoy_types

    def generate_decoys(self, target_df) -> pl.DataFrame:

        if self.method is None:
            return None

        if self.method == "swap":
            decoys, decoy_types = self._swap_decoys(target_df["peptide"].to_list())
            return target_df.with_columns(
                pl.Series("peptide", decoys, dtype=pl.String),
                pl.Series("decoy_type", decoy_types, dtype=pl.String),
            ).filter(pl.col("peptide").is_not_null())

        if self.method == "pseudo_reverse":
            get_decoy = (
                pl.col("peptide").str.slice(0, 1)
                + pl.col("peptide")
                .str.slice(1, pl.col("sequence_length") - 1)
                .str.reverse()
                + pl.col("peptide").str.slice(pl.col("sequence_length"), 2)
            ).alias("peptide")
            decoy_df = target_df.with_columns(get_decoy)

        elif self.method == "mutation":
            get_decoy = (
                pl.col("peptide").map_elements(
                    partial(get_mutated_decoy, varmod_signature=self._varmod_signature),
                    return_dtype=pl.String,
                )
            ).alias("peptide")
            decoy_df = target_df.with_columns(get_decoy)

        elif self.method == "pseudo_shuffle":
            random.seed(self.random_seed)
            get_decoy = (
                pl.col("peptide").map_elements(
                    get_shuffled_decoy, return_dtype=pl.String
                )
            ).alias("peptide")
            decoy_df = target_df.with_columns(get_decoy)

        else:
            raise NotImplementedError(
                f"Decoy generation method {self.method} is not implemented"
            )

        return decoy_df.with_columns(decoy_type=pl.lit(self.method))

    def resolve_duplicate_decoys(
        self, target_peptide_df, decoy_df, max_attempts: int = 3
    ) -> pl.DataFrame:
        """Regenerate decoys shared by several targets.

        ``target_peptide_df`` and ``decoy_df`` are matched row by row. All but
        the first occurrence of a duplicate decoy are replaced by a
        pseudo-shuffle of their target (terminal residues kept). Decoys still
        colliding after ``max_attempts`` are collapsed into one.
        """

        random.seed(self.random_seed)

        target_peptides = target_peptide_df["peptide"].to_list()
        decoy_peptides = decoy_df["peptide"].to_list()

        for _ in range(max_attempts):
            # rows duplicating an earlier decoy
            dup_indices = (
                pl.DataFrame({"peptide": decoy_peptides})
                .with_row_index("__row")
                .filter(pl.int_range(pl.len()).over("peptide") > 0)
                .get_column("__row")
                .to_list()
            )
            if not dup_indices:
                break

            for i in dup_indices:
                decoy_peptides[i] = get_shuffled_decoy(target_peptides[i])

        decoy_df = decoy_df.with_columns(pl.Series("peptide", decoy_peptides))

        return decoy_df.unique(subset="peptide", keep="first", maintain_order=True)

    def append_decoys(
        self,
        target_peptide_df,
    ) -> pl.DataFrame:

        decoy_peptide_df = self.generate_decoys(
            target_peptide_df.with_row_index("target_peptide_index")
        )
        if decoy_peptide_df is None:
            return target_peptide_df.with_columns(is_decoy=False)

        # re-group decoys by peptide and collect protein indices for each decoy
        decoy_peptide_df = (
            # protein_index is never an empty list, so empty_as_null is a no-op here
            decoy_peptide_df.explode("protein_index", empty_as_null=True)
            .group_by("peptide", maintain_order=True)
            .agg(
                pl.col("protein_index").unique().sort(),
                pl.col("sequence_length").first(),
                # explode() repeats each target once per protein -> dedupe
                pl.col("target_peptide_index").unique().sort(),
                pl.col("decoy_type").first(),
            )
        )
        # remove decoys spectrally identical to a target (markers ignored, I == L)
        decoy_peptide_df = (
            decoy_peptide_df.with_columns(il_normalized_sequence_expr().alias("_key"))
            .join(
                target_peptide_df.select(il_normalized_sequence_expr().alias("_key")),
                on="_key",
                how="anti",
            )
            .drop("_key")
        )

        # target columns, then the new ones
        out_cols = list(target_peptide_df.columns) + [
            "is_decoy",
            "target_peptide_index",
            "decoy_type",
        ]

        target_out = target_peptide_df.with_columns(
            is_decoy=False,
            # with_row_index uses UInt64 on rt64; keep its aggregated list dtype.
            target_peptide_index=pl.lit(
                None, dtype=decoy_peptide_df.schema["target_peptide_index"]
            ),
            decoy_type=pl.lit(None, dtype=pl.String),
        ).select(out_cols)
        decoy_out = decoy_peptide_df.with_columns(is_decoy=True).select(out_cols)

        return pl.concat((target_out, decoy_out), how="vertical")
