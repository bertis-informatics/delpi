from enum import Enum, IntEnum


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


class IonType(IntEnum):
    PRECURSOR = 0
    B = 1
    Y = 2


class AcquisitionMode(IntEnum):
    DDA = 0
    DIA = 1


class TheoPeakInput(InputEnum):
    PRED_INTENSITY = "predicted_intensity", 0, "float32"
    ION_TYPE = "ion_type", 1, "uint8"
    NEUTRAL_LOSS = "neutral_loss", 2, "uint8"
    CHARGE = "charge", 3, "uint8"
    ISOTOPE_INDEX = "isotope_index", 4, "int8"
    CLEAVAGE_INDEX = "cleavage_index", 5, "uint8"
    REV_CLEAVAGE_INDEX = "reverse_cleavage_index", 6, "uint8"


class ExpPeakInput(InputEnum):
    MZ_ERROR = "mz_error", 0, "float32"
    AB = "ab", 1, "float32"
    Z_SCORE = "z_score", 2, "float32"
    MS_LEVEL = "ms_level", 3, "uint8"
    ION_TYPE = "ion_type", 4, "uint8"
    NEUTRAL_LOSS = "neutral_loss", 5, "uint8"
    CHARGE = "charge", 6, "uint8"
    ISOTOPE_INDEX = "isotope_index", 7, "int8"
    TIME_INDEX = "time_index", 8, "uint8"
    CLEAVAGE_INDEX = "cleavage_index", 9, "uint8"
    REV_CLEAVAGE_INDEX = "reverse_cleavage_index", 10, "uint8"


class ContextInput(InputEnum):
    ACQUISITION_MODE = "acquisition_mode", 0, "uint8"
    NCE = "nce", 1, "float32"
    LIBRARY_NCE = "library_nce", 2, "float32"
    ISOLATION_LEFT = "isolation_left", 3, "float32"
    ISOLATION_RIGHT = "isolation_right", 4, "float32"


THEORETICAL_PEAK = TheoPeakInput.field_names()
EXPERIMENTAL_PEAK = ExpPeakInput.field_names()
CONTEXT = ContextInput.field_names()
