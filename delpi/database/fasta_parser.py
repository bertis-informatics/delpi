from pathlib import Path
import re

import polars as pl
from fastaparser import Reader

from delpi.database.enzyme import Enzyme


class FastaParser:
    """Read protein sequences from FASTA file"""

    def __init__(self, fasta_path: str):
        self.fasta_path = Path(fasta_path)

    @property
    def fasta_name(self):
        return self.fasta_path.stem

    def parse(self) -> pl.DataFrame:
        def remove_c_term_x(seq):
            return seq[:-1] if seq.endswith("X") else seq

        data = []
        with open(self.fasta_path, "r") as fasta:
            reader = Reader(fasta, parse_method="quick")
            data = [(seq.header, remove_c_term_x(seq.sequence)) for seq in reader]

        seq_df = (
            pl.DataFrame(
                data, schema={"header": pl.String, "sequence": pl.String}, orient="row"
            )
            .with_columns(pl.col("header").str.split(" ").alias("header_splitted"))
            .with_columns(
                pl.col("header_splitted").list.first().str.slice(1).alias("fasta_id")
            )
            .with_columns(
                pl.col("header_splitted")
                .list.slice(1)
                .list.join(" ")
                .alias("description")
            )
            .select(pl.col("fasta_id", "description", "sequence"))
        )

        return seq_df

    @staticmethod
    def generate_decoy_sequence_df(
        target_df: pl.DataFrame, enzyme: str, fasta_id_prefix: str = "rev_"
    ) -> pl.DataFrame:

        pattern = re.compile(Enzyme.name_to_pattern[enzyme])

        def _pseudo_reverse(sequence):
            cutpos = (
                [0]
                + [m.start() + 1 for m in pattern.finditer(sequence)]
                + [len(sequence)]
            )
            decoy_sequence = list(sequence)
            for start, end in zip(cutpos[:-1], cutpos[1:]):
                if end - start < 3:
                    continue
                decoy_sequence[start : end - 1] = reversed(
                    decoy_sequence[start : end - 1]
                )
            return "".join(decoy_sequence)

        decoy_df = target_df.with_columns(
            pl.col("sequence").map_elements(_pseudo_reverse, return_dtype=pl.String)
        ).with_columns((fasta_id_prefix + pl.col("fasta_id")).alias("fasta_id"))

        return decoy_df
