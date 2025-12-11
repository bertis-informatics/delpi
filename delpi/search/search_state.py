from enum import IntEnum


class SearchState(IntEnum):
    INIT = 0
    DB_PREP = 1

    FIRST_SEARCH = 2
    FIRST_TDA = 3
    FIRST_TL_PREP = 4

    TL_TRAINING = 5
    REFINED_DB_PREP = 6

    SECOND_SEARCH = 7
    SECOND_TDA = 8

    QUANTIFICATION = 9
    DONE = 10
