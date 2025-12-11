import re

from .composition import Composition
from .unimod_db import UniModDatabase

UNKNOWN_PSI_MOD_ACCESSION = 1001460
MOD_MASS_FORMAT_STRING = '{:.3f}'


class Modification:

    spectronaut_mod_pattern = re.compile(r'^[A-Z][a-z]+ \(.+\)$')
    name_to_mod_map = dict()
    mass_to_mod_map = dict()
    accession_num_to_mod_map = dict()

    def __init__(self, accession_num, composition, name, description=None) -> None:

        assert isinstance(composition, Composition)
        self.accession_num = accession_num
        self.composition = composition
        self.name = name # psi-ms-name
        self.description = description

    def __hash__(self) -> int:
        return self.accession_num if self.accession_num > 0 else hash(self.name)

    @property
    def mass(self):
        return self.composition.mass


    def __eq__(self, other):
        return (
            isinstance(other, Modification) and 
            self.accession_num == other.accession_num and 
            self.composition == other.composition
        )

    def __str__(self) -> str:
        return self.name


    @classmethod
    def get(cls, psi_ms_name):
        if psi_ms_name is None:
            return None

        mod_name = psi_ms_name.lower()
        if mod_name.startswith('unimod:'):
            accession_num = int(mod_name[len('unimod:'):])
            return cls.accession_num_to_mod_map[accession_num]
        else:
            #if mod_name in cls.name_to_mod_map:
            return cls.name_to_mod_map[mod_name]
        
    @classmethod
    def get_by_unimod_id(cls, unimod_id):
        return cls.accession_num_to_mod_map[unimod_id]

    @classmethod
    def get_from_mass(cls, delta_mass_str):
        delta_mass = MOD_MASS_FORMAT_STRING.format(float(delta_mass_str))
        return Modification.mass_to_mod_map[delta_mass][0]
    
    @classmethod
    def register(cls, modification):
        mod_name = modification.name.lower()
        cls.name_to_mod_map[mod_name] = modification
        cls.accession_num_to_mod_map[modification.accession_num] = modification

        mass_str = MOD_MASS_FORMAT_STRING.format(modification.composition.mass)
        if mass_str not in cls.mass_to_mod_map:
            cls.mass_to_mod_map[mass_str] = []
        cls.mass_to_mod_map[mass_str].append(modification)

    @classmethod
    def find_name_from_mass(cls, square_bracket_mass):
        try:
            mod = cls.get_from_mass(square_bracket_mass)
        except:
            raise NotImplementedError(f'Cannot parse delta mass [{square_bracket_mass}]')
        return mod.name
    
    @classmethod
    def get_max_accession_num(cls):
        return max(list(cls.accession_num_to_mod_map))
    


# Load and register modifications stored in unimod DB
from delpi.chem.unimod_db import UniModDatabase

unimod_db = UniModDatabase()
for mod_record in unimod_db.get_modifications():
    try:
        mod = Modification(
            accession_num=mod_record['accession_num'],
            name=mod_record['name'],
            description=mod_record['description'],
            composition=Composition.parse(mod_record['composition'])
        )
        Modification.register(mod)
    except Exception as e:
        print(e)
