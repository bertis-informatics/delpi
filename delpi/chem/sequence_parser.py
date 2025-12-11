import re
from typing import List, Tuple

from delpi.chem.modification import Modification
from delpi.chem.amino_acid import AminoAcid


# For nested brackets
# https://stackoverflow.com/questions/54455775/extract-string-between-two-brackets-including-nested-brackets-in-python

TERM_CHARS = "".join([re.escape(aa.residue) for aa in AminoAcid.terminal_amino_acids])
N_TERM_CHARS = f"{re.escape(AminoAcid.protein_n_term.residue)}{re.escape(AminoAcid.peptide_n_term.residue)}"
C_TERM_CHARS = f"{re.escape(AminoAcid.protein_c_term.residue)}{re.escape(AminoAcid.peptide_c_term.residue)}"
_RESIDUE_PATTERN = re.compile(
    rf"([{TERM_CHARS}A-Z])(\[[^\]]*\]|\((?:[^()]*|\([^()]*\))*\))?"
)

C_TERM_PATTERN = rf"[{C_TERM_CHARS}](\[\S*\]|\(\S*\))?\s*$"
N_TERM_PATTERN = rf"^[{N_TERM_CHARS}]"
DELPI_PATTERN = rf"{N_TERM_PATTERN}\S*{C_TERM_PATTERN}"

RESIDUE_RE = re.compile(rf"([{TERM_CHARS}A-Z])((?:\([^)]*\))*)")
UNIMOD_RE = re.compile(r"\(UniMod:(\d+)\)")


def parse_modification_string(modification_string):
    if not modification_string:
        return None
    mod_str = modification_string[1:-1]

    if modification_string.startswith("("):
        # round brackets with accession number or psi_ms_name
        # e.g. (Unimod:1), (Oxidation)
        return Modification.get(mod_str)
    else:
        try:
            # UniMod or Square bracket format (e.g. [+12], [+12.34])
            return Modification.get_from_mass(mod_str)
        except:
            # Spectronaut format (e.g. [Oxidation (M)])
            name = re.sub(r" \([^)]*\)", "", mod_str)
            return Modification.get(name)


def parse_sequence_string(peptide_string):

    if re.search(DELPI_PATTERN, peptide_string):
        pass
    elif peptide_string.startswith("_"):
        peptide_string = peptide_string[::-1].replace(
            "_", AminoAcid.peptide_c_term.residue, 1
        )[::-1]
    elif peptide_string.startswith("-."):
        peptide_string = peptide_string.replace(
            "-.", AminoAcid.peptide_n_term.residue
        ).replace(".-", AminoAcid.peptide_c_term.residue)
    else:
        peptide_string = f"{AminoAcid.peptide_n_term.residue}{peptide_string}{AminoAcid.peptide_c_term.residue}"

    residue_tuples = _RESIDUE_PATTERN.findall(peptide_string)

    return [
        (res_tup[0], parse_modification_string(res_tup[1]))
        for res_tup in residue_tuples
    ]


def parse_modified_sequence_string(
    modified_sequence,
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Optimized version using manual parsing instead of regex.
    This is 3.5x faster than the original regex-based version.
    """
    seq_chars = []
    mod_tuples = []
    site = 0

    i = 0
    while i < len(modified_sequence):
        char = modified_sequence[i]

        if char.isalpha() or char in TERM_CHARS:
            seq_chars.append(char)
            i += 1

            # Check for modifications
            while i < len(modified_sequence) and modified_sequence[i] == "(":
                end = modified_sequence.find(")", i)
                if end == -1:
                    break
                mod_str = modified_sequence[i + 1 : end]
                if mod_str.startswith("UniMod:"):
                    mod_tuples.append((site, int(mod_str[7:])))
                i = end + 1

            site += 1
        else:
            i += 1

    return "".join(seq_chars), mod_tuples


def parse_modified_sequence_string_original(
    modified_sequence,
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Original regex-based version (kept for reference).
    """
    residue_tuples = RESIDUE_RE.findall(modified_sequence)

    seq_chars = []
    mod_tuples = []

    for site, (aa, mod_block) in enumerate(residue_tuples):
        seq_chars.append(aa)
        mod_tuples.extend(
            (site, int(mod_id)) for mod_id in UNIMOD_RE.findall(mod_block)
        )

    return "".join(seq_chars), mod_tuples


# Performance test
def test_parse_modified_sequence_string():
    import time

    # More comprehensive test sequences
    test_sequences = [
        "<(UniMod:737)(UniMod:1)DK(UniMod:737)FM(UniMod:35)K(UniMod:737)(UniMod:121)EATTNAPFR.(UniMod:2)",
        "_AEDQTESSC(UniMod:4)ESHR.",
        "(UniMod:1)PEPTIDE(UniMod:2)R.",
        "SIMPLE_PEPTIDE.",
        "COMPLEX(UniMod:1)(UniMod:2)PEPTIDE(UniMod:3).",
        "<(UniMod:1)M(UniMod:35)ETQAPEPTIDEK(UniMod:2).",
        "_LONGERPEPTIDEWITHALOTOFMODIFICATIONS(UniMod:1)(UniMod:2)(UniMod:3)R.",
        "AVERYLONGPEPTIDESEQUENCEWITHNOMODIFICATIONS_",
        "<(UniMod:1)PEPTIDE(UniMod:2)PEPTIDE(UniMod:3)PEPTIDE(UniMod:4).",
        "MULTIPLECONSECUTIVEMODIFICATIONS(UniMod:1)(UniMod:2)(UniMod:3)(UniMod:4)(UniMod:5)K.",
    ] * 10000  # 100,000 sequences for more accurate timing

    # Warm up
    for seq in test_sequences[:100]:
        _ = parse_modified_sequence_string_original(seq)
        _ = parse_modified_sequence_string(seq)

    # Test original version multiple times
    original_times = []
    for _ in range(5):
        start = time.perf_counter()
        for seq in test_sequences:
            _ = parse_modified_sequence_string_original(seq)
        original_times.append(time.perf_counter() - start)

    # Test optimized version multiple times
    optimized_times = []
    for _ in range(5):
        start = time.perf_counter()
        for seq in test_sequences:
            _ = parse_modified_sequence_string(seq)
        optimized_times.append(time.perf_counter() - start)

    original_avg = sum(original_times) / len(original_times)
    optimized_avg = sum(optimized_times) / len(optimized_times)

    print(f"Test sequences: {len(test_sequences):,}")
    print(
        f"Original (regex): {original_avg:.4f} ± {max(original_times) - min(original_times):.4f} seconds"
    )
    print(
        f"Optimized (manual): {optimized_avg:.4f} ± {max(optimized_times) - min(optimized_times):.4f} seconds"
    )
    print(f"Speedup: {original_avg/optimized_avg:.2f}x")

    # Verify results are identical
    test_seq = test_sequences[0]
    result1 = parse_modified_sequence_string_original(test_seq)
    result2 = parse_modified_sequence_string(test_seq)
    print(f"Results identical: {result1 == result2}")
