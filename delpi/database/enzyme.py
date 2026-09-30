import multiprocessing as mp

import polars as pl

from delpi.chem.amino_acid import AminoAcid

PROTEIN_N_TERM = AminoAcid.protein_n_term.residue
PROTEIN_C_TERM = AminoAcid.protein_c_term.residue
PEPTIDE_N_TERM = AminoAcid.peptide_n_term.residue
PEPTIDE_C_TERM = AminoAcid.peptide_c_term.residue


class Enzyme:
    """Protein digestion rules.

    Cleavage sites are computed directly from three residue sets rather than
    a combined regex pattern:

    - ``cut_after``: cleave immediately after any of these residues.
    - ``cut_before``: cleave immediately before any of these residues.
    - ``exclude_before``: suppress a cleavage site when the residue right
      after the site is one of these (e.g. the trypsin/proline exception:
      ``cut_after=[K, R], exclude_before=[P]``).

    A predefined enzyme ``name`` (see ``name_to_rule``) is just a preset
    that is expanded into the same three sets; ``name`` and custom rules are
    mutually exclusive.
    """

    # Legacy combined regex patterns. Kept only for
    # FastaParser.generate_decoy_sequence_df (pseudo-reverse decoy
    # generation), which is independent of the cleavage-rule digestion below.
    name_to_pattern = {
        "trypsin": r"([KR])",
        "chymotrypsin": r"([FLY](?=[^P]))|(W(?=[^MP]))|(M(?=[^PY]))|(H(?=[^DMPW]))",
        "lys-c": "K",
        "glu-c": "E",
        "asp-n": r"\w(?=D)",
    }

    # Predefined enzymes expressed as (cut_after, cut_before, exclude_before).
    name_to_rule = {
        "trypsin": (("K", "R"), (), ()),
        "lys-c": (("K",), (), ()),
        "glu-c": (("E",), (), ()),
        "asp-n": ((), ("D",), ()),
    }

    def __init__(
        self,
        name=None,
        cut_after=None,
        cut_before=None,
        exclude_before=None,
        min_len=7,
        max_len=30,
        n_term_methionine_excision=True,
        max_missed_cleavages=1,
    ):

        has_custom_rule = any(
            arg is not None for arg in (cut_after, cut_before, exclude_before)
        )
        if name is not None and has_custom_rule:
            raise ValueError(
                "Specify either a predefined enzyme 'name' or custom cleavage "
                "rules (cut_after/cut_before/exclude_before), not both."
            )

        if has_custom_rule:
            self.name = None
        else:
            name = name or "trypsin"
            if name not in self.name_to_rule:
                raise ValueError(
                    f"Unknown enzyme '{name}'. Supported presets: "
                    f"{sorted(self.name_to_rule)}. Alternatively, specify "
                    "custom cut_after/cut_before/exclude_before rules."
                )
            cut_after, cut_before, exclude_before = self.name_to_rule[name]
            self.name = name

        self.cut_after = self._validate_residues(cut_after, "cut_after")
        self.cut_before = self._validate_residues(cut_before, "cut_before")
        self.exclude_before = self._validate_residues(exclude_before, "exclude_before")

        if not self.cut_after and not self.cut_before:
            raise ValueError(
                "At least one of 'cut_after' or 'cut_before' must be non-empty."
            )

        self.min_len = min_len
        self.max_len = max_len
        self.n_term_methionine_excision = n_term_methionine_excision
        self.max_missed_cleavages = max_missed_cleavages

    @staticmethod
    def _validate_residues(residues, field_name):
        if not residues:
            return frozenset()
        invalid = [
            r
            for r in residues
            if not (
                isinstance(r, str) and len(r) == 1 and AminoAcid.is_standard_residue(r)
            )
        ]
        if invalid:
            raise ValueError(
                f"Invalid residue(s) in '{field_name}': {invalid}. Must be "
                f"single-letter standard amino acid codes: "
                f"{AminoAcid.standard_amino_acid_chars}"
            )
        return frozenset(residues)

    @property
    def param_dict(self):
        param_dict = {
            "min_len": self.min_len,
            "max_len": self.max_len,
            "max_missed_cleavages": self.max_missed_cleavages,
            "n_term_methionine_excision": self.n_term_methionine_excision,
        }
        if self.name is not None:
            param_dict["enzyme"] = self.name
        else:
            param_dict["cut_after"] = sorted(self.cut_after)
            param_dict["cut_before"] = sorted(self.cut_before)
            param_dict["exclude_before"] = sorted(self.exclude_before)
        return param_dict

    @staticmethod
    def pad_peptide_terminals(seq):
        if seq[0] == PROTEIN_N_TERM and seq[-1] == PROTEIN_C_TERM:
            return seq
        elif seq[0] == PROTEIN_N_TERM:
            return seq + PEPTIDE_C_TERM
        elif seq[-1] == PROTEIN_C_TERM:
            return PEPTIDE_N_TERM + seq
        return PEPTIDE_N_TERM + seq + PEPTIDE_C_TERM

    def cleavage_positions(self, protein_sequence):
        """Sorted, de-duplicated indices right before which a cleavage occurs.

        A position ``p`` splits the sequence as ``seq[:p] | seq[p:]``. Only
        internal positions (``0 < p < len(seq)``) are considered.
        """

        seq_len = len(protein_sequence)
        positions = set()

        if self.cut_after:
            positions.update(
                i + 1
                for i in range(seq_len - 1)
                if protein_sequence[i] in self.cut_after
            )
        if self.cut_before:
            positions.update(
                i for i in range(1, seq_len) if protein_sequence[i] in self.cut_before
            )
        if self.exclude_before:
            positions = {
                p for p in positions if protein_sequence[p] not in self.exclude_before
            }

        return sorted(positions)

    def digest_protein(self, protein_sequence):

        max_missed_cleavages = self.max_missed_cleavages
        seq_len = len(protein_sequence)

        cutpos = [0] + self.cleavage_positions(protein_sequence) + [seq_len]
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
                digest_results = pool.map(self.digest_protein, sequence_df["sequence"])
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
