import re
from .unimod_db import UniModDatabase


class Atom(object):
    atom_map = dict()
    H, C, N, O, P, S = (None, )*6

    def __init__(self, code, mono_mass, avg_mass, name):
        self.nominal_mass = round(mono_mass)
        self.code = code
        self.mono_mass = mono_mass
        self.avg_mass = avg_mass
        self.name = name

    @property
    def mass(self):
        return self.mono_mass

    @staticmethod
    def get(code):
        return Atom.atom_map[code]

    def __eq__(self, other):
        if self is other:
            return True
        return self.name == other.name

    def __hash__(self):
        return hash(self.code)
    
    def get_isotope_string(self):
        if self.code[0].isdigit():
            match = re.search(r"(\d+)(\S+)", self.code)
            isotope = int(match.group(1))
            element_ = match.group(2)
            return f'{element_}[{isotope}]' 
        else:
            return self.code


unimod_db = UniModDatabase()
Atom.atom_map = {
    atom['code']: Atom(
                    code=atom['code'],
                    name=atom['name'],
                    mono_mass=float(atom['mono_mass']),
                    avg_mass=float(atom['avg_mass'])
                ) for atom in unimod_db.get_atoms()
}

# shared global variales
Atom.H = Atom.get('H')
Atom.C = Atom.get('C')
Atom.N = Atom.get('N')
Atom.O = Atom.get('O')
Atom.P = Atom.get('P')
Atom.S = Atom.get('S')

