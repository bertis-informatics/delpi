from itertools import accumulate
import operator

import numpy as np

from delpi.chem.composition import Composition
from delpi.chem.amino_acid import AminoAcid
from delpi.chem.modified_amino_acid import ModifiedAminoAcid
from delpi.chem.cleavage import Cleavage, Fragment
from delpi.chem.sequence_parser import parse_sequence_string
from delpi.constants import PROTON_MASS


class Sequence:

    def __init__(self, amino_acids):

        assert amino_acids[0].residue in [AminoAcid.protein_n_term.residue, AminoAcid.peptide_n_term.residue]
        assert amino_acids[-1].residue in [AminoAcid.protein_c_term.residue, AminoAcid.peptide_c_term.residue]
        
        self.amino_acids = list(amino_acids)
        self._prefix_compositions = None
        self._prefix_mass_array = None
    
    @property
    def prefix_mass_array(self):
        if self._prefix_mass_array is None:
            mass_arr = np.array(list(accumulate(aa.mass for aa in self.amino_acids[:-1]))[1:], dtype=np.float64)
            mass_arr[-1] += self.amino_acids[-1].mass
            self._prefix_mass_array = mass_arr

        return self._prefix_mass_array
    
    @property
    def composition(self):
        return self.prefix_compositions[-1]
    
    @property
    def prefix_compositions(self):

        if self._prefix_compositions is None:
            composition_list = [aa.composition for aa in self.amino_acids[:-1]]
            prefix_compositions = list(accumulate(composition_list, operator.add))[1:]
            # add c-term composition to the last residue's composition
            prefix_compositions[-1] += self.amino_acids[-1].composition
            self._prefix_compositions = prefix_compositions
            
        return self._prefix_compositions

    def __getitem__(self, index):
        return self.amino_acids[index+1]

    def get_stripped_sequence(self):
        return self.get_plain_string()

    def get_plain_string(self):
        return "".join([aa.residue for aa in self.amino_acids[1:-1]])

    def get_modified_sequence(self):
        return ''.join([str(aa) for aa in self.amino_acids])
    
    def __str__(self):
        return self.to_string()

    def to_string(self, with_modification):
        return self.get_modified_sequence() if with_modification else self.get_stripped_sequence()

    @classmethod
    def from_string(cls, sequence_str):
        """
        https://pyopenms.readthedocs.io/en/latest/user_guide/peptides_proteins.html
        Args:
            sequence_str (str): sequence string with specified modifications
        """
        # residues, n_term_mod, c_term_mod = parse_sequence(sequence_str)
        # parsed_residues = parse_sequence_string('.(Dimethyl)DFPIAMGER.')
        parsed_residues = parse_sequence_string(sequence_str)

        # Parsed residues may contain protein terminal characters. 
        # Let's use peptide characters only from here:
        amino_acids = [
                ModifiedAminoAcid.get(AminoAcid.get(residue), modification) 
                    for residue, modification in parsed_residues
            ]
        # amino_acids = [ModifiedAminoAcid.get(AminoAcid.peptide_n_term, parsed_residues[0][1])]
        # amino_acids.extend([ModifiedAminoAcid.get(AminoAcid.get(residue), modification) 
        #                     for residue, modification in parsed_residues[1:-1]])
        # amino_acids.append(ModifiedAminoAcid.get(AminoAcid.peptide_c_term, parsed_residues[-1][1]))
        
        return cls(amino_acids)


    def __len__(self):
        return len(self.amino_acids) - 2

    # def internal_cleavages(self):
    #     for i in range(len(self) - 1):
    #         yield Cleavage(self, i)

    def internal_cleavages(self):
        seq_len = len(self)
        total_mass = self.prefix_mass_array[-1]
        for i in range(seq_len - 1):
            prefix_frag_mass = self.prefix_mass_array[i]
            suffix_frag_mass = total_mass-prefix_frag_mass
            yield Cleavage(
                    prefix=Fragment(self, i, True, prefix_frag_mass), 
                    suffix=Fragment(self, i, False, suffix_frag_mass))

    def get_prefix_compositions(self):
        return self.prefix_compositions

    def get_suffix_compositions(self):
        return [self.composition - comp for comp in self.prefix_compositions[::-1]]

    def get_precursor_mass(self):
        return self.prefix_mass_array[-1] + Composition.H2O.mass

    def get_precursor_composition(self):
        return self.composition + Composition.H2O
    
    def get_precursor_mz(self, charge):
        return self.get_precursor_mass()/charge + PROTON_MASS

    def has_n_terminal_modification(self):
        return isinstance(self.amino_acids[0], ModifiedAminoAcid)

    def has_c_terminal_modification(self):
        return isinstance(self.amino_acids[-1], ModifiedAminoAcid)
    
    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for a1, a2 in zip(self.amino_acids, other.amino_acids):
            if a1 != a2:
                return False
        return True

    def __hash__(self):
        return hash(self.to_string())
        
        