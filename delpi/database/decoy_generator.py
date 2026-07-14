import random

import polars as pl

from delpi.database.numba.prefix_mass_array import aa_mass_array

MUTATION_MAP = dict(zip("GAVLIFMPWSCTYHKRQEND", "LLLVVLLLLTSSSSLLNDQE"))


def get_mutated_decoy(peptide):
    return (
        peptide[:2]
        + MUTATION_MAP[peptide[2]]
        + peptide[3:-3]
        + MUTATION_MAP[peptide[-3]]
        + peptide[-2:]
    )


def _diann_fragment_shifts(peptide: str):
    """Return (delta_N, delta_C) monoisotopic mass shifts for a diann-style decoy.

    Replicates the two mutation positions used by DIA-NN without altering the
    actual sequence:
    - ``peptide[2]``  (second residue)   → N-side shift applied from b2 onward
    - ``peptide[-3]`` (second-to-last)   → C-side shift applied from the
      second-to-last prefix mass onward
    """
    delta_n = (
        aa_mass_array[ord(MUTATION_MAP[peptide[2]])] - aa_mass_array[ord(peptide[2])]
    )
    delta_c = (
        aa_mass_array[ord(MUTATION_MAP[peptide[-3]])] - aa_mass_array[ord(peptide[-3])]
    )
    return float(delta_n), float(delta_c)


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

    def __init__(self, method: str = None, random_seed: int = 323):

        if method is not None and method not in self.supported_methods:
            raise ValueError(
                f"Unsupported decoy generation method: {method}. "
                f"Supported methods: {self.supported_methods}"
            )
        self.method = method
        self.random_seed = random_seed

    def generate_decoys(self, peptide_df) -> pl.DataFrame:

        if self.method is None:
            return None

        target_df = peptide_df.select(pl.exclude("peptide_index"))

        if self.method == "pseudo_reverse":
            get_decoy = (
                pl.col("peptide").str.slice(0, 1)
                + pl.col("peptide")
                .str.slice(1, pl.col("sequence_length") - 1)
                .str.reverse()
                + pl.col("peptide").str.slice(pl.col("sequence_length"), 2)
            ).alias("peptide")
            decoy_df = target_df.with_columns(get_decoy)

        # elif self.method == "mutation_legacy":
        #     get_decoy = (
        #         pl.col("peptide").map_elements(
        #             get_mutated_decoy, return_dtype=pl.String
        #         )
        #     ).alias("peptide")
        #     decoy_df = target_df.with_columns(get_decoy)

        elif self.method in ["diann", "mutation"]:
            # Sequence is kept identical to the target.  Only fragment m/z will
            # differ via per-peptide shifts stored as metadata columns.
            peptides = peptide_df["peptide"].to_list()
            shifts = [_diann_fragment_shifts(p) for p in peptides]
            decoy_df = target_df.with_columns(
                pl.Series(
                    "decoy_n_fragment_shift", [s[0] for s in shifts], dtype=pl.Float32
                ),
                pl.Series(
                    "decoy_c_fragment_shift", [s[1] for s in shifts], dtype=pl.Float32
                ),
            )

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

        decoy_peptide_df = self.generate_decoys(target_peptide_df)

        _zero_shifts = [
            pl.lit(0.0).cast(pl.Float32).alias("decoy_n_fragment_shift"),
            pl.lit(0.0).cast(pl.Float32).alias("decoy_c_fragment_shift"),
        ]

        if decoy_peptide_df is None:
            return target_peptide_df.with_columns(is_decoy=False, *_zero_shifts)

        if self.method == "diann":
            # diann decoys share the target sequence; no anti-join needed.
            decoy_peptide_df = (
                decoy_peptide_df.explode("protein_index")
                .group_by("peptide", maintain_order=True)
                .agg(
                    pl.col("protein_index").unique().sort(),
                    pl.col("sequence_length").first(),
                    pl.col("decoy_n_fragment_shift").first(),
                    pl.col("decoy_c_fragment_shift").first(),
                )
            )
        else:
            # re-group decoys by peptide and collect protein indices for each decoy
            decoy_peptide_df = (
                decoy_peptide_df.explode("protein_index")
                .group_by("peptide", maintain_order=True)
                .agg(
                    pl.col("protein_index").unique().sort(),
                    pl.col("sequence_length").first(),
                )
            )
            # remove decoys that are identical to any target peptide
            decoy_peptide_df = decoy_peptide_df.join(
                target_peptide_df.select(pl.col("peptide")), on="peptide", how="anti"
            )
            decoy_peptide_df = decoy_peptide_df.with_columns(*_zero_shifts)

        # Determine output column order: original target columns then new ones.
        out_cols = list(target_peptide_df.columns) + [
            "is_decoy",
            "decoy_n_fragment_shift",
            "decoy_c_fragment_shift",
        ]
        target_out = target_peptide_df.with_columns(
            is_decoy=False, *_zero_shifts
        ).select(out_cols)
        decoy_out = decoy_peptide_df.with_columns(is_decoy=True).select(out_cols)

        return pl.concat((target_out, decoy_out), how="vertical")
