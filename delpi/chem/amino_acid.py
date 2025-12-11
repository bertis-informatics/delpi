import numpy as np

from .composition import Composition


class AminoAcid:
    protein_n_term = None
    protein_c_term = None
    peptide_n_term = None
    peptide_c_term = None

    standard_amino_acids = list()
    terminal_amino_acids = list()
    standard_amino_acid_chars = None
    standard_amino_acid_map = dict()
    all_amino_acid_array = np.empty(128, dtype=np.object_)

    def __init__(self, residue, name, composition) -> None:

        self.residue = residue
        self.name = name
        self._composition = composition

    @property
    def composition(self):
        return self._composition

    @property
    def mass(self):
        return self._composition.mass

    @property
    def nominal_mass(self):
        return self._composition.nominal_mass

    @staticmethod
    def is_standard_residue(residue):
        return residue in AminoAcid.standard_amino_acid_chars

    def __str__(self) -> str:
        return self.residue

    @property
    def is_modified(self):
        return False

    @classmethod
    def get(cls, residue):
        return cls.all_amino_acid_array[ord(residue)]

    def __eq__(self, other):
        if (type(self) is type(other)) and self.residue == other.residue:
            return False
        return False

    def __hash__(self) -> int:
        return hash(self.residue)


AminoAcid.protein_n_term = AminoAcid("<", "Protein-N-terminus", Composition.ZERO)
AminoAcid.protein_c_term = AminoAcid(">", "Protein-C-terminus", Composition.ZERO)
AminoAcid.peptide_n_term = AminoAcid("_", "Peptide-N-terminus", Composition.ZERO)
AminoAcid.peptide_c_term = AminoAcid(".", "Peptide-C-terminus", Composition.ZERO)

AminoAcid.terminal_amino_acids = [
    AminoAcid.protein_n_term,
    AminoAcid.protein_c_term,
    AminoAcid.peptide_n_term,
    AminoAcid.peptide_c_term,
]

AminoAcid.standard_amino_acids = [
    AminoAcid("G", "Glycine", Composition(2, 3, 1, 1, 0)),
    AminoAcid("A", "Alanine", Composition(3, 5, 1, 1, 0)),
    AminoAcid("S", "Serine", Composition(3, 5, 1, 2, 0)),
    AminoAcid("P", "Proline", Composition(5, 7, 1, 1, 0)),
    AminoAcid("V", "Valine", Composition(5, 9, 1, 1, 0)),
    AminoAcid("T", "Threonine", Composition(4, 7, 1, 2, 0)),
    AminoAcid("C", "Cysteine", Composition(3, 5, 1, 1, 1)),
    AminoAcid("L", "Leucine", Composition(6, 11, 1, 1, 0)),
    AminoAcid("I", "Isoleucine", Composition(6, 11, 1, 1, 0)),
    AminoAcid("N", "Asparagine", Composition(4, 6, 2, 2, 0)),
    AminoAcid("D", "Aspartate", Composition(4, 5, 1, 3, 0)),
    AminoAcid("Q", "Glutamine", Composition(5, 8, 2, 2, 0)),
    AminoAcid("K", "Lysine", Composition(6, 12, 2, 1, 0)),
    AminoAcid("E", "Glutamate", Composition(5, 7, 1, 3, 0)),
    AminoAcid("M", "Methionine", Composition(5, 9, 1, 1, 1)),
    AminoAcid("H", "Histidine", Composition(6, 7, 3, 1, 0)),
    AminoAcid("F", "Phenylalanine", Composition(9, 9, 1, 1, 0)),
    AminoAcid("R", "Arginine", Composition(6, 12, 4, 1, 0)),
    AminoAcid("Y", "Tyrosine", Composition(9, 9, 1, 2, 0)),
    AminoAcid("W", "Tryptophan", Composition(11, 10, 2, 1, 0)),
    # Non-standard amino acids (https://www.cup.uni-muenchen.de/ch/compchem/tink/as.html)
    # AminoAcid('U', "Selenocysteine", Composition(3, 5, 1, 1, 0, additional_elements={Atom.get('Se'): 1})),
    # AminoAcid('O', "Pyrrolysine", Composition(12, 19, 3, 2, 0)),
    # AminoAcid('O', "Hydroxyproline", Composition(5, 7, 1, 2, 0)),
]

for aa in AminoAcid.standard_amino_acids + AminoAcid.terminal_amino_acids:
    AminoAcid.all_amino_acid_array[ord(aa.residue)] = aa

AminoAcid.standard_amino_acid_map = {
    aa.residue: aa for aa in AminoAcid.standard_amino_acids
}

AminoAcid.standard_amino_acid_chars = "".join(
    [aa.residue for aa in AminoAcid.standard_amino_acids]
)
