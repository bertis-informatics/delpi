import re
from enum import Enum

from delpi.chem.modification import Modification
from delpi.chem.amino_acid import AminoAcid

MOD_SEPARATOR = ";"
PROT_N_TERM_AA = AminoAcid.protein_n_term.residue
PROT_C_TERM_AA = AminoAcid.protein_c_term.residue
PEPT_N_TERM_AA = AminoAcid.peptide_n_term.residue
PEPT_C_TERM_AA = AminoAcid.peptide_c_term.residue


class ModificationLocation(Enum):
    ANYWHERE = "anywhere", 0
    PEPTIDE_N_TERM = "peptide_n_term", 1
    PEPTIDE_C_TERM = "peptide_c_term", 2
    PROTEIN_N_TERM = "protein_n_term", 3
    PROTEIN_C_TERM = "protein_c_term", 4

    def __new__(cls, value, index):
        member = object.__new__(cls)
        member._value_ = value
        member.index = index
        return member

    def __int__(self):
        return self.index


INDEX_TO_LOC = {e.index: e for e in ModificationLocation}

get_mod_loc_index = lambda mod_param_id: (mod_param_id >> 13) & 0x7
get_mod_loc = lambda mod_param_id: INDEX_TO_LOC[get_mod_loc_index(mod_param_id)]
get_unimod_id = lambda mod_param_id: (mod_param_id >> 16) & 0xFFFF


def _encode(
    unimod_id: int, loc: ModificationLocation, residue: str, fixed: bool
) -> int:

    # unimod_id = self.modification.accession_num # 0-65535 (16 bit)
    # loc = self.location # 0-8       (3 bit)
    # residue = self.residue # 0-128  (7 bit)
    # fixed = self.fixed # 0-1        (1 bit)
    if not (0 <= unimod_id <= 65535):
        raise ValueError("unimod_id must be in the range 0-65535")
    if not (0 <= int(loc) <= 7):
        raise ValueError("Index of location index must be in the range 0-7")
    if len(residue) != 1 or not (0 <= ord(residue) <= 127):
        raise ValueError("residue must be a single ASCII character")

    encoded = (unimod_id << 16) | (int(loc) << 13) | (ord(residue) << 6) | int(fixed)
    return encoded


class ModificationParam:

    def __init__(
        self,
        mod_name: str | Modification,
        residue: str,
        location: str | ModificationLocation,
        fixed: bool,
    ):

        assert isinstance(residue, str) and len(residue) == 1

        self.modification = (
            mod_name
            if isinstance(mod_name, Modification)
            else Modification.get(mod_name)
        )
        self.location = (
            location
            if isinstance(location, ModificationLocation)
            else ModificationLocation(location.lower())
        )
        self.residue = residue
        self.fixed = fixed

    @property
    def mod_name(self):
        return self.modification.name

    @classmethod
    def decode(cls, mod_param_id: int):
        unimod_id = (mod_param_id >> 16) & 0xFFFF
        loc_index = (mod_param_id >> 13) & 0x7
        residue = chr((mod_param_id >> 6) & 0x7F)
        fixed = bool(mod_param_id & 0x1)

        return cls(
            mod_name=Modification.get_by_unimod_id(unimod_id),
            residue=residue,
            location=INDEX_TO_LOC[loc_index],
            fixed=fixed,
        )

    def encode(self) -> int:
        return _encode(
            unimod_id=self.modification.accession_num,
            loc=self.location,
            residue=self.residue,
            fixed=self.fixed,
        )

    def __hash__(self):
        return self.encode()

    @property
    def id(self):
        return self.encode()

    @property
    def unimod_id(self):
        return self.modification.accession_num

    def __str__(self):
        mod_type = "static" if self.fixed else "variable"
        return f"{self.modification.name}, {self.residue}, {self.location.value}, {mod_type}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__str__()})"

    def get_re_pattern(self):

        if self.location == ModificationLocation.ANYWHERE:
            pattern = f"{self.residue}"
        else:
            residue = "" if self.residue == "*" else self.residue
            if self.location == ModificationLocation.PROTEIN_N_TERM:
                pattern = f"^{re.escape(PROT_N_TERM_AA)}{residue}"
            elif self.location == ModificationLocation.PROTEIN_C_TERM:
                pattern = f"{residue}{re.escape(PROT_C_TERM_AA)}$"
            elif self.location == ModificationLocation.PEPTIDE_N_TERM:
                pattern = f"^[{re.escape(PROT_N_TERM_AA)}{re.escape(PEPT_N_TERM_AA)}]{residue}"
            elif self.location == ModificationLocation.PEPTIDE_C_TERM:
                pattern = f"{residue}[{re.escape(PROT_C_TERM_AA)}{re.escape(PEPT_C_TERM_AA)}]$"

        return pattern

    def match_groups(self):
        if self.residue == "*":
            if self.location == ModificationLocation.PROTEIN_N_TERM:
                return [PROT_N_TERM_AA]
            if self.location == ModificationLocation.PROTEIN_C_TERM:
                return [PROT_C_TERM_AA]
            if self.location == ModificationLocation.PEPTIDE_N_TERM:
                return [PEPT_N_TERM_AA, PROT_N_TERM_AA]
            if self.location == ModificationLocation.PEPTIDE_C_TERM:
                return [PEPT_C_TERM_AA, PROT_C_TERM_AA]
            raise NotImplementedError()
        else:
            return [self.residue]

    def to_dict(self):
        return {
            # 'mod_id': self.id,
            # 'unimod_id': self.unimod_id,
            "mod_name": self.modification.name,
            "residue": self.residue,
            "location": self.location.value,
            "fixed": self.fixed,
        }


def get_test_mod_param_set():

    # mod_params = ModificationParamSet()
    # mod_params.add('Carbamidomethyl', 'C', ModificationLocation.ANYWHERE, True)
    # mod_params.add('Label:13C(6)15N(2)', 'K', ModificationLocation.PEPTIDE_C_TERM, True)
    # mod_params.add('Label:13C(6)15N(4)', 'R', ModificationLocation.PEPTIDE_C_TERM, True)
    # mod_params.add('Oxidation', 'M', ModificationLocation.ANYWHERE, False)
    # mod_params.add('Acetyl', '*', ModificationLocation.PROTEIN_N_TERM, False)
    # mod_params.add('Amidated', '*', ModificationLocation.PROTEIN_C_TERM, False)

    mod_parms = [
        ("Carbamidomethyl", "C", "anywhere", True),
        ("Label:13C(6)15N(2)", "K", "peptide_c_term", True),
        ("Label:13C(6)15N(4)", "R", "peptide_c_term", True),
        ("Oxidation", "M", "anywhere", False),
        ("Acetyl", "*", "protein_n_term", False),
        ("Amidated", "*", "protein_c_term", False),
    ]

    mod_param_set = [ModificationParam(*p) for p in mod_parms]

    return mod_param_set
