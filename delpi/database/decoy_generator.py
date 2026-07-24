import random
from functools import partial
from typing import Callable, Optional, Sequence

import polars as pl

MUTATION_MAP = dict(zip("GAVLIFMPWSCTYHKRQEND", "LLLVVLLLLTSSSSLLNDQE"))


def _default_varmod_signature(aa: str) -> tuple:
    """Trivial signature used when no variable modifications are active:
    every residue compares equal, so every position is considered safe."""
    return ()


def build_varmod_signature_fn(
    mod_param_set: Optional[Sequence],
) -> Callable[[str], tuple]:
    """Build a function mapping a residue character to an eligibility tuple
    over the active (non-fixed) residue-specific variable modifications.

    Residues are grouped by modification name so that e.g. Phospho(S)/(T)/(Y)
    collapse into a single "is this residue phospho-eligible" flag rather
    than three independent ones (matching the ``generate_variable_modifications``
    residue-matching logic in :class:`~delpi.database.modification_handler.ModificationHandler`).
    Two residues are considered "safe" to swap for decoy mutation when they
    share the same signature tuple. Fixed modifications are ignored.
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
    """Try ``candidates`` (0-based indices; negative values count from the
    end, as in Python slicing) in priority order and return the first
    internal position (never the first or last residue, and never
    ``exclude``) whose ``MUTATION_MAP`` mutation preserves the residue's
    variable-modification eligibility. Returns ``None`` if none qualifies."""
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
    """Mutate two internal residues of ``peptide`` to generate a decoy,
    preserving the first residue and the last two residues.

    Each terminal side tries a fixed, prioritized list of positions and
    mutates the first one whose ``MUTATION_MAP`` substitution preserves the
    residue's eligibility for any active variable modification (see
    :func:`build_varmod_signature_fn`):

    - N-terminal side: index 2, then 3, then 4, then 1.
    - C-terminal side: index ``len - 3``, then ``len - 4``, then ``len - 5``.

    If none of a side's candidates are safe (or valid), that side falls back
    to its original default position (2 for N-term, ``len(peptide) - 3`` for
    C-term) and mutates it unconditionally, so the decoy always differs from
    the target. The C-terminal search skips any position already claimed by
    the N-terminal side.
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


class DecoyGenerator:

    supported_methods = [
        "pseudo_reverse",
        "mutation",
        "pseudo_shuffle",
        "diann",
    ]

    def __init__(
        self,
        method: str = None,
        random_seed: int = 323,
        mod_param_set: Optional[Sequence] = None,
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

    def generate_decoys(self, target_df) -> pl.DataFrame:

        if self.method is None:
            return None

        if self.method == "pseudo_reverse":
            get_decoy = (
                pl.col("peptide").str.slice(0, 1)
                + pl.col("peptide")
                .str.slice(1, pl.col("sequence_length") - 1)
                .str.reverse()
                + pl.col("peptide").str.slice(pl.col("sequence_length"), 2)
            ).alias("peptide")
            decoy_df = target_df.with_columns(get_decoy)

        elif self.method in ("diann", "mutation"):
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

        return decoy_df

    def resolve_duplicate_decoys(
        self, target_peptide_df, decoy_df, max_attempts: int = 3
    ) -> pl.DataFrame:
        """Resolve decoys shared by multiple targets.

        Different target sequences can map to the same decoy sequence. For each
        group of colliding decoys the first occurrence is kept, while the rest
        are regenerated with a pseudo-shuffle that preserves the N-term and
        C-term residues of the corresponding target and only shuffles the
        interior. ``target_peptide_df`` and ``decoy_df`` are matched row by row.
        Decoys that still collide after ``max_attempts`` are collapsed into a
        single decoy.
        """

        random.seed(self.random_seed)

        target_peptides = target_peptide_df["peptide"].to_list()
        decoy_peptides = decoy_df["peptide"].to_list()

        for _ in range(max_attempts):
            # Indices of rows that duplicate an earlier decoy (keep the first).
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
            decoy_peptide_df.explode("protein_index")
            .group_by("peptide", maintain_order=True)
            .agg(
                pl.col("protein_index").unique().sort(),
                pl.col("sequence_length").first(),
                pl.col("target_peptide_index"),
            )
        )
        # remove decoys that are identical to any target peptide
        decoy_peptide_df = decoy_peptide_df.join(
            target_peptide_df.select(pl.col("peptide")), on="peptide", how="anti"
        )

        # Determine output column order: original target columns then new ones.
        out_cols = list(target_peptide_df.columns) + [
            "is_decoy",
            "target_peptide_index",
        ]

        target_out = target_peptide_df.with_columns(
            is_decoy=False, target_peptide_index=pl.lit(None, dtype=pl.List(pl.UInt32))
        ).select(out_cols)
        decoy_out = decoy_peptide_df.with_columns(is_decoy=True).select(out_cols)

        return pl.concat((target_out, decoy_out), how="vertical")
