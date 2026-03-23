import random

import polars as pl

MUTATION_MAP = dict(zip("GAVLIFMPWSCTYHKRQEND", "LLLVVLLLLTSSSSLLNDQE"))


def get_mutated_decoy(peptide):
    return (
        peptide[:2]
        + MUTATION_MAP[peptide[2]]
        + peptide[3:-3]
        + MUTATION_MAP[peptide[-3]]
        + peptide[-2:]
    )


def get_shuffled_decoy(peptide):
    mid = list(peptide[1:-2])
    random.shuffle(mid)
    return f"{peptide[0]}{''.join(mid)}{peptide[-2:]}"


class DecoyGenerator:

    def __init__(self, method: str = None, random_seed: int = 323):

        assert method in [
            "pseudo_reverse",
            "pseudo_shuffle",
            "diann",
            None,
        ], "`pseudo_reverse`, `pseudo_shuffle`, and `diann` methods are supported"

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

        elif self.method == "diann":
            get_decoy = (
                pl.col("peptide").map_elements(
                    get_mutated_decoy, return_dtype=pl.String
                )
            ).alias("peptide")
        elif self.method == "pseudo_shuffle":
            random.seed(self.random_seed)
            get_decoy = (
                pl.col("peptide").map_elements(
                    get_shuffled_decoy, return_dtype=pl.String
                )
            ).alias("peptide")
        else:
            raise NotImplementedError(
                f"Decoy generation method {self.method} is not implemented"
            )

        decoy_df = target_df.with_columns(get_decoy)

        return decoy_df

    def append_decoys(
        self,
        target_peptide_df,
        allow_conflicts: bool = False,
    ) -> pl.DataFrame:

        decoy_peptide_df = self.generate_decoys(target_peptide_df)
        if decoy_peptide_df is None:
            return target_peptide_df.with_columns(is_decoy=False)

        if allow_conflicts == False:
            decoy_peptide_df = decoy_peptide_df.join(
                target_peptide_df.select(pl.col("peptide")), on="peptide", how="anti"
            )

        return pl.concat(
            (
                target_peptide_df.with_columns(is_decoy=False),
                decoy_peptide_df.with_columns(is_decoy=True),
            ),
            how="vertical",
        )
