from enum import Enum


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


class TheoPeakInput(InputEnum):
    PRED_INTENSITY = "predicted_intensity", 0, "float32"
    MZ = "mz", 1, "float32"

    IS_PRECURSOR = "is_precursor", 2, "bool"
    IS_PREFIX = "is_prefix", 3, "bool"
    CHARGE = "charge", 4, "uint8"
    ISOTOPE_INDEX = "isotope_index", 5, "uint8"

    # 2D positional encoding
    CLEAVAGE_INDEX = "cleavage_index", 6, "uint8"
    REV_CLEAVAGE_INDEX = "reverse_cleavage_index", 7, "uint8"


class ExpPeakInput(InputEnum):
    MZ_ERROR = "mz_error", 0, "float32"
    AB = "ab", 1, "float32"
    Z_SCORE = "z_score", 2, "float32"

    MS_LEVEL = "ms_level", 3, "uint8"
    IS_PRECURSOR = "is_precursor", 4, "bool"
    IS_PREFIX = "is_prefix", 5, "bool"
    CHARGE = "charge", 6, "uint8"
    ISOTOPE_INDEX = "isotope_index", 7, "uint8"

    # 1D positional encoding
    TIME_INDEX = "time_index", 8, "uint8"

    # 2D positional encoding
    CLEAVAGE_INDEX = "cleavage_index", 9, "uint8"
    REV_CLEAVAGE_INDEX = "reverse_cleavage_index", 10, "uint8"


THEORETICAL_PEAK = TheoPeakInput.field_names()
EXPERIMENTAL_PEAK = ExpPeakInput.field_names()
