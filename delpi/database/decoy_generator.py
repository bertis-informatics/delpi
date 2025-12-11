import polars as pl

DIANN_MAP = {aa: ma for aa, ma in zip("GAVLIFMPWSCTYHKRQEND", "LLLVVLLLLTSSSSLLNDQE")}


def get_diann_decoy(peptide):
    return (
        peptide[:2]
        + DIANN_MAP[peptide[2]]
        + peptide[3:-3]
        + DIANN_MAP[peptide[-3]]
        + peptide[-2:]
    )


class DecoyGenerator:

    def __init__(self, method: str = None):

        assert method in [
            "pseudo_reverse",
            "diann",
            None,
        ], "`pseudo_reverse` and `diann` methods are supported"

        self.method = method

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
                pl.col("peptide").map_elements(get_diann_decoy, return_dtype=pl.String)
            ).alias("peptide")

        decoy_df = target_df.with_columns(get_decoy)

        return decoy_df

    def append_decoys(self, target_peptide_df) -> pl.DataFrame:

        decoy_peptide_df = self.generate_decoys(target_peptide_df)

        if decoy_peptide_df is None:
            return target_peptide_df.with_columns(is_decoy=False)

        return pl.concat(
            (
                target_peptide_df.with_columns(is_decoy=False),
                decoy_peptide_df.with_columns(is_decoy=True),
            ),
            how="vertical",
        )
