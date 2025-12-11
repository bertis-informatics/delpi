import re

from .atom import Atom
from ..constants import C13C12_MASS_DIFF


class Composition:
    ZERO, H2O, NH3, NH2, OH, CO, H = (None, )*7

    def __init__(self, c, h, n, o, s, p=0, additional_elements=None) -> None:
        super().__init__()
        self._c = c
        self._h = h
        self._n = n
        self._o = o
        self._s = s
        self._p = p
        if additional_elements is None:
            self.additional_elements = dict() # (atom, count)
        else:
            self.additional_elements = additional_elements.copy()
        self._mass = None
        self._nominal_mass = None

    def _reset_mass(self):
        self._mass = None
        self._nominal_mass = None

    def get_elements(self):
        elements = self.base_elements()
        elements.update(
            {atom.code: cnt for atom, cnt in self.additional_elements.items()})
        return elements

    @property
    def mass(self):
        if self._mass is None:
            self._mass = self.get_monoisotopic_mass()
        return self._mass

    @property
    def nominal_mass(self):
        if self._nominal_mass is None:
            self._nominal_mass = self.get_nominal_mass()
        return self._nominal_mass

    def __hash__(self) -> int:
        return hash(self.mass)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if (self._c != other._c | 
            self._h != other._h | 
            self._n != other._n | 
            self._o != other._o | 
            self._p != other._p | 
            self._s != other._s):
            return False

        if len(self.additional_elements) != len(other.additional_elements):
            return False

        for elem, count in other.additional_elements.items():
            if elem not in self.additional_elements:
                return False
            if self.additional_elements[elem] != count:
                return False

        return True

    @staticmethod
    def merge_additional_elements(elem1, elem2):
        additional_elements = elem1.copy()
        for elem, count in elem2.items():
            if elem in additional_elements:
                additional_elements[elem] += count
            else:
                additional_elements[elem] = count        
        return additional_elements
        
    def _add_composition(self, other):

        assert type(other) == self.__class__

        # merge additional elements
        additional_elements = self.merge_additional_elements(
            self.additional_elements, other.additional_elements
        )
        
        # return a new composition
        return Composition(
                c = self._c + other._c,
                h = self._h + other._h,
                n = self._n + other._n,
                o = self._o + other._o,
                p = self._p + other._p,
                s = self._s + other._s,
                additional_elements=additional_elements
            )
        return self

    def _add(self, other):
        if type(other) != Composition:
            return other._add(self)

        return self._add_composition(other)

    def __neg__(self):
        return Composition(
            c=-self._c,
            h=-self._h,
            n=-self._n,
            o=-self._o,
            p=-self._p,
            s=-self._s,
            additional_elements={
                elem: -count for elem, count in self.additional_elements.items()
            }
        )

    def __add__(self, other):
        if type(self) != Composition:
            return self._add(other)

        return other._add(self)

    def __sub__(self, other):
        return self + (-other)


    def get_nominal_mass(self):
        mass = self._c * Atom.C.nominal_mass + \
                self._h * Atom.H.nominal_mass + \
                self._n * Atom.N.nominal_mass + \
                self._o * Atom.O.nominal_mass + \
                self._s * Atom.S.nominal_mass + \
                self._p * Atom.P.nominal_mass
        
        mass += sum([atom_.nominal_mass*count for atom_, count in self.additional_elements.items()])
        return mass

    def get_monoisotopic_mass(self):
        mass = self._c * Atom.C.mass + \
                self._h * Atom.H.mass + \
                self._n * Atom.N.mass + \
                self._o * Atom.O.mass + \
                self._s * Atom.S.mass + \
                self._p * Atom.P.mass
        
        mass += sum([atom_.mass*count for atom_, count in self.additional_elements.items()])
        return mass

    def base_elements(self):
        return {
                'C': self._c,
                'H': self._h,
                'N': self._n,
                'O': self._o,
                'S': self._s,
                'P': self._p,
            }
    
    def _compute_isotope_envelope(self, max_num_isotopes, right_tail_relative_intensity_threshold):
        # elements = self.base_elements()
        # elements.update({atom_.get_isotope_string(): count for atom_, count in self.additional_elements.items()})
        # return IsotopomerEnvelope.create(
        #             elements, 
        #             max_num_isotopes=max_num_isotopes, 
        #             right_tail_relative_intensity_threshold=right_tail_relative_intensity_threshold)
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.to_string()

    def to_string(self):
        composition_list = [
                f'{code}({count})' 
                    for code, count in self.base_elements().items() 
                        if code != 'P' or count > 0]

        composition_list.extend(
                [f'{atom_.code}({count})' 
                        for atom_, count in self.additional_elements.items()])
        
        return ' '.join(composition_list)

    def to_plain_string(self):
        composition_list = [
                f'{code}{count}' 
                    for code, count in self.base_elements().items() 
                        if count > 0]
        composition_list.extend(
                [f'{atom_.code}{count}' 
                        for atom_, count in self.additional_elements.items()])
        return ''.join(composition_list)
    

    @staticmethod
    def parse(composition_str):
        c, h, n, o, s, p = 0, 0, 0, 0, 0, 0
        additional_elems = dict()
        for token in composition_str.split():
            if re.match(r"^\d*[a-zA-Z]+(\(-?\d+\))?$", token) is None:
                raise ValueError('Composition string in wrong format')

            if re.match(r"^\d*?[a-zA-Z]+$", token) is not None:
                element = token
                count = 1
            else:
                m = re.search(r'\((-?\d+(?:\.\d+)?)\)', token)
                element = token[:m.start()]
                count = int(token[m.start()+1:m.end()-1])
                if element == 'C':
                    c += count
                elif element == 'H':
                    h += count
                elif element == 'N':
                    n += count
                elif element == 'O':
                    o += count
                elif element == 'S':
                    s += count
                elif element == 'P':
                    p += count
                else:
                    atom_ = Atom.get(element)
                    if atom_ not in additional_elems:
                        additional_elems[atom_] = 0
                        
                    additional_elems[atom_] += count

        return Composition(c, h, n, o, s, p, additional_elements=additional_elems)
    
    def get_isotope_mass(self, isotope_index):
        return self.mass + isotope_index * C13C12_MASS_DIFF

    def get_isotopomer_envelope(self, max_num_isotopes=16, right_tail_relative_intensity_threshold=0.1):
        # return self._compute_isotope_envelope(max_num_isotopes, right_tail_relative_intensity_threshold)
        raise NotImplementedError()

    def get_most_abundant_isotope_index(self):
        # envelope = self.get_isotopomer_envelope()
        # return envelope.most_abundant_isotope_index
        raise NotImplementedError()


    @staticmethod
    def parse_from_plain_string(formula_str):

        #formula_str = 'C2H3N1O1S'
        if re.match(r"^([A-Z][a-z]?-?\d*)+$", formula_str) is None:
            raise ValueError('plain-string empirical formula in wrong format')

        matches = re.findall(r"[A-Z][a-z]?-?\d*", formula_str)
        composition_str = []
        for elem in matches:
            atom = re.match("[A-Z][a-z]?", elem)
            count = elem[atom.end():]
            count = int(count) if len(count) > 0 else 1
            composition_str.append(f'{atom.group()}({count})')

        composition_str = ' '.join(composition_str)

        return Composition.parse(composition_str)
    

# initialize shared global variables in composition
Composition.ZERO = Composition(0, 0, 0, 0, 0)
Composition.H2O = Composition(0, 2, 0, 1, 0)
Composition.NH3 = Composition(0, 3, 1, 0, 0)
Composition.NH2 = Composition(0, 2, 1, 0, 0)
Composition.OH = Composition(0, 1, 0, 1, 0)
Composition.CO = Composition(1, 0, 0, 1, 0)
Composition.H = Composition(0, 1, 0, 0, 0)