from enum import Enum, IntEnum

INPUT_SCHEMA_VERSION = 4


class InputEnum(Enum):

    def __new__(cls, value: str, index: int, dtype: str):
        member = object.__new__(cls)
        member._value_ = value
        member.index = index
        member.dtype = dtype
        return member

    def __int__(self):
        return self.index

    def __str__(self):
        return self._value_

    @classmethod
    def field_names(cls):
        return [e.value for e in cls]


class AcquisitionMode(IntEnum):
    DDA = 0
    DIA = 1


class TheoPeakInput(InputEnum):
    PRED_INTENSITY = "predicted_intensity", 0, "float32"
    IS_PRECURSOR = "is_precursor", 1, "bool"
    IS_PREFIX = "is_prefix", 2, "bool"
    IS_MODLOSS = "is_modloss", 3, "bool"  # False: NoLoss, True: H3O4P
    CHARGE = "charge", 4, "uint8"
    ISOTOPE_INDEX = "isotope_index", 5, "int8"
    CLEAVAGE_INDEX = "cleavage_index", 6, "uint8"
    REV_CLEAVAGE_INDEX = "reverse_cleavage_index", 7, "uint8"


class ExpPeakInput(InputEnum):
    MZ_ERROR = "mz_error", 0, "float32"
    AB = "ab", 1, "float32"
    Z_SCORE = "z_score", 2, "float32"
    MS_LEVEL = "ms_level", 3, "uint8"
    IS_PRECURSOR = "is_precursor", 4, "bool"
    IS_PREFIX = "is_prefix", 5, "bool"
    IS_MODLOSS = "is_modloss", 6, "bool"  # False: NoLoss, True: H3O4P
    CHARGE = "charge", 7, "uint8"
    ISOTOPE_INDEX = "isotope_index", 8, "int8"
    TIME_INDEX = "time_index", 9, "int8"  # Scan offset from the RT center (0).
    CLEAVAGE_INDEX = "cleavage_index", 10, "uint8"
    REV_CLEAVAGE_INDEX = "reverse_cleavage_index", 11, "uint8"


class ContextInput(InputEnum):
    ACQUISITION_MODE = "acquisition_mode", 0, "uint8"
    NCE = "nce", 1, "float32"
    LIBRARY_NCE = "library_nce", 2, "float32"
    ISOLATION_LEFT = "isolation_left", 3, "float32"
    ISOLATION_RIGHT = "isolation_right", 4, "float32"


THEORETICAL_PEAK = TheoPeakInput.field_names()
EXPERIMENTAL_PEAK = ExpPeakInput.field_names()
CONTEXT = ContextInput.field_names()
