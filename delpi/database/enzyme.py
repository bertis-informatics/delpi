import re
import multiprocessing as mp

import polars as pl

from delpi.chem.amino_acid import AminoAcid

PROTEIN_N_TERM = AminoAcid.protein_n_term.residue
PROTEIN_C_TERM = AminoAcid.protein_c_term.residue
PEPTIDE_N_TERM = AminoAcid.peptide_n_term.residue
PEPTIDE_C_TERM = AminoAcid.peptide_c_term.residue


class Enzyme:

    name_to_pattern = {
        "trypsin": r"([KR])",
        "chymotrypsin": r"([FLY](?=[^P]))|(W(?=[^MP]))|(M(?=[^PY]))|(H(?=[^DMPW]))",
        "lys-c": "K",
        "glu-c": "E",
        "asp-n": r"\w(?=D)",
    }

    def __init__(
        self,
        name="trypsin",
        min_len=7,
        max_len=30,
        n_term_methionine_excision=True,
        max_missed_cleavages=1,
    ):

        assert name in self.name_to_pattern

        self.name = name
        self.min_len = min_len
        self.max_len = max_len
        self.n_term_methionine_excision = n_term_methionine_excision
        self.cleavage_pattern = re.compile(self.name_to_pattern[name])
        self.max_missed_cleavages = max_missed_cleavages

    @property
    def param_dict(self):
        return {
            "enzyme": self.name,
            "min_len": self.min_len,
            "max_len": self.max_len,
            "max_missed_cleavages": self.max_missed_cleavages,
            "n_term_methionine_excision": self.n_term_methionine_excision,
        }

    @staticmethod
    def pad_peptide_terminals(seq):
        if seq[0] == PROTEIN_N_TERM and seq[-1] == PROTEIN_C_TERM:
            return seq
        elif seq[0] == PROTEIN_N_TERM:
            return seq + PEPTIDE_C_TERM
        elif seq[-1] == PROTEIN_C_TERM:
            return PEPTIDE_N_TERM + seq
        return PEPTIDE_N_TERM + seq + PEPTIDE_C_TERM

    def digest_protein(self, protein_sequence):

        pattern = self.cleavage_pattern
        max_missed_cleavages = self.max_missed_cleavages
        seq_len = len(protein_sequence)

        # Search with endpos. because there could be a zero-length peptide, when sequence ends with 'K' or 'R'
        cutpos = (
            [0]
            + [
                m.start() + 1
                for m in pattern.finditer(protein_sequence, endpos=seq_len - 1)
            ]
            + [seq_len]
        )
        peptides = [
            protein_sequence[cutpos[i] : cutpos[i + 1]] for i in range(len(cutpos) - 1)
        ]
        num_peptides = len(peptides)

        # attach protein terminal residue characters
        peptides[0] = f"{PROTEIN_N_TERM}{peptides[0]}"
        peptides[-1] = f"{peptides[-1]}{PROTEIN_C_TERM}"

        # Missed cleavages
        for num_missed in range(1, max_missed_cleavages + 1):
            peptides.extend(
                [
                    "".join(peptides[k : k + num_missed + 1])
                    for k in range(num_peptides - num_missed)
                ]
            )

        # N-terminal methionine excision
        if self.n_term_methionine_excision and protein_sequence[0] == "M":
            nme_peptides = peptides[: max_missed_cleavages + 1]
            nme_peptides[0] = f"{PROTEIN_N_TERM}{nme_peptides[0][2:]}"
            peptides.extend(
                ["".join(nme_peptides[:k]) for k in range(1, len(nme_peptides) + 1)]
            )

        peptides = set(peptides)
        peptides = [self.pad_peptide_terminals(seq) for seq in peptides]
        peptides = [
            pep
            for pep in peptides
            if len(pep) > self.min_len + 1 and len(pep) < self.max_len + 3
        ]

        return peptides

    def digest(
        self,
        sequence_df: pl.DataFrame,
        use_multiprocessing: bool = False,
    ) -> pl.DataFrame:

        n_proc = mp.cpu_count() // 2
        if use_multiprocessing and n_proc > 1 and sequence_df.shape[0] > 10000:
            from delpi.utils.mp import get_multiprocessing_context
            # Use 'spawn' context to avoid deadlocks with multi-threaded environments
            with get_multiprocessing_context().Pool(processes=n_proc) as pool:
                digest_results = pool.map(
                    self.digest_protein, sequence_df["sequence"]
                )
        else:
            digest_results = [self.digest_protein(s) for s in sequence_df["sequence"]]

        tmp = [
            (i, pep)
            for i, peps in enumerate(digest_results)
            if len(peps) > 0
            for pep in peps
        ]
        peptide_df = pl.DataFrame({"peptide": [t[1] for t in tmp]})
        seq_indices = [t[0] for t in tmp]
        peptide_df = peptide_df.with_columns(
            pl.Series(values=seq_indices, name="protein_index", dtype=pl.UInt32)
        )

        peptide_df = (
            peptide_df.sort(pl.col("peptide", "protein_index"))
            .group_by(["peptide"], maintain_order=True)
            .agg(pl.col("protein_index"))
            .with_columns(
                # (pl.col('peptide').str.head(1) == PROT_N_TERM_AA).alias('protein_n_term'),
                # (pl.col('peptide').str.tail(1) == PROT_C_TERM_AA).alias('protein_c_term'),
                (pl.col("peptide").str.len_chars() - 2)
                .cast(pl.UInt16)
                .alias("sequence_length")
            )
        )

        return peptide_df
