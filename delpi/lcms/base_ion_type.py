from delpi.chem.composition import Composition

# from delpi.constants import PROTON_MASS

# def default_composition_calculator(possible_off_sets, amino_acid):

#     if amino_acid is None:
#         return [CompositionWithDeltaMass(off) for off in possible_off_sets['*']]

#     key = amino_acid.residue if amino_acid.residue in possible_off_sets else '*'
#     return [CompositionWithDeltaMass(off) - amino_acid.composition for off in possible_off_sets[key]]


class BaseIonType:

    # public delegate IEnumerable<Composition> CompositionCalculator(AminoAcid aminoAcid = null);
    A, Ar, B, C, D, V, W, X, Xr, Y, YM1, Zr, Z = (None,) * 13
    base_ion_type_map = dict()

    def __init__(
        self, symbol, is_prefix, offset_composition, composition_calculator=None
    ) -> None:

        self.symbol = symbol
        self.is_prefix = is_prefix
        self.offset_composition = offset_composition

        if composition_calculator is not None:
            self._composition_calculator = composition_calculator
        else:
            self._composition_calculator = lambda amino_acid: [self.offset_composition]

    @property
    def is_suffix(self):
        return not self.is_prefix

    def __str__(self):
        return self.symbol

    def __hash__(self):
        return hash(self.symbol)

    def __eq__(self, other):
        if isinstance(other, BaseIonType):
            return self.symbol == other.symbol
        return False

    def get_possible_compositions(self, amino_acid):
        return self._composition_calculator(amino_acid)

    @classmethod
    def get_all_base_ion_types(cls):
        return [
            attr
            for attr in BaseIonType.__dict__.values()
            if isinstance(attr, BaseIonType)
        ]

    @classmethod
    def get(cls, symbol):
        return cls.base_ion_type_map[symbol.lower()]


# For A, B, C, X, Y, Z ions: sum of all amino acids + offset
BaseIonType.A = BaseIonType("a", True, Composition(-1, 0, 0, -1, 0))
BaseIonType.Ar = BaseIonType(
    "a.", True, BaseIonType.A.offset_composition + Composition.H
)
BaseIonType.B = BaseIonType("b", True, Composition.ZERO)
BaseIonType.C = BaseIonType("c", True, Composition.NH3)

# BaseIonType.X = BaseIonType("x", False, CompositionWithDeltaMass(44.9977) - Composition.H)
# BaseIonType.Xr = BaseIonType("x.", False, BaseIonType.X.offset_composition + Composition.H)
BaseIonType.Y = BaseIonType("y", False, Composition.H2O)
BaseIonType.YM1 = BaseIonType(
    "y-1", False, BaseIonType.Y.offset_composition - Composition.H
)

BaseIonType.Z = BaseIonType("z", False, Composition.H2O - Composition.NH2)
BaseIonType.Zr = BaseIonType(
    "z.", False, BaseIonType.Z.offset_composition + Composition.H
)

# register base ion types to dictionary
BaseIonType.base_ion_type_map = {
    v.symbol: v for k, v in BaseIonType.__dict__.items() if isinstance(v, BaseIonType)
}

# # D ions have additional options for isoleucine and threonine.
# # All offsets defined as sum of previous residues + offset.
# _d_ion_offsets = {
#     # Defined in terms of sum of previous residue weights + offset
#     '*': [ 44.0500 - Composition.H.mass ], # For all residues except for V, I and T
#     'V': [ 58.0657 - Composition.H.mass ],
#     'I': [ 58.0657 - Composition.H.mass, 72.0813 - Composition.H.mass ], # for isoleucine
#     'T': [ 58.0657 - Composition.H.mass, 60.0450 - Composition.H.mass ],  # for threonine
# }
# BaseIonType.D = BaseIonType("d", True, CompositionWithDeltaMass(44.0500),
#                     lambda amino_acid: default_composition_calculator(_d_ion_offsets, amino_acid))

# # V only has one option for all terminal residues. Sum of previous residues + offset
# BaseIonType.V = BaseIonType("v", False, CompositionWithDeltaMass(74.0242),
#                     lambda amino_acid:
#                         CompositionWithDeltaMass(74.0242) if amino_acid is None else
#                         CompositionWithDeltaMass(74.0242) - amino_acid.composition - Composition.H)

# # W ions have additional options for isoleucine and threonine.
# # All offsets defined as sum of previous residues + offset.
# # Defined in terms of sum of previous residue weights + offset
# w_ion_offsets = {
#     '*': [ 73.0290 - Composition.H.mass ], # For all residues except for V, I and T
#     'V': [ 87.0446 - Composition.H.mass ],
#     'I': [ 87.0446 - Composition.H.mass, AminoAcid.get('I').mass - 12.0238 - Composition.H.mass ], # for isoleucine
#     'T': [ 87.0446 - Composition.H.mass, AminoAcid.get('T').mass - 12.0238 - Composition.H.mass ]  # for threonine
# }
# BaseIonType.W = BaseIonType("w", False, CompositionWithDeltaMass(73.0290),
#                   lambda amino_acid: default_composition_calculator(w_ion_offsets, amino_acid))
