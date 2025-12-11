from .modification import Modification
from .amino_acid import AminoAcid


class ModifiedAminoAcid(AminoAcid):

    cached_modified_amino_aicds = dict()

    def __init__(self, amino_acid, modification) -> None:
        super().__init__(
                amino_acid.residue, 
                amino_acid.name, 
                amino_acid.composition + modification.composition)
        
        # assert not isinstance(amino_acid, self.__class__)
        assert type(amino_acid) == AminoAcid, "Only AminoAcid instances can have modification"
        assert isinstance(modification, Modification)
        self.modification = modification

    def __eq__(self, other):
        if super.__eq__(other) and (self.modification == other.modification):
            return True
        return False       

    def __hash__(self):
        return hash(self.residue) ^ hash(self.modification)
    
    def __str__(self) -> str:
        return f'{self.residue}(UniMod:{self.modification.accession_num})'

    @property
    def is_modified(self):
        return True

    @classmethod
    def get(cls, amino_acid, modification):
        
        if modification is None:
            return amino_acid

        key = (amino_acid, modification)
        maa = cls.cached_modified_amino_aicds.get(key)

        if maa is None:
            maa = cls(*key)
            cls.cached_modified_amino_aicds[key] = maa
        
        return maa