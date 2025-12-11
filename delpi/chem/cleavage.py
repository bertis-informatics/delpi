from typing import NamedTuple


class Fragment(NamedTuple):
    sequence: object  # Sequence object
    cleavage_index: int
    is_prefix: bool
    mass: float

    @property
    def index(self):
        if self.is_prefix:
            return self.cleavage_index + 1
        return len(self.sequence) - self.cleavage_index - 1

    def get_composition(self):
        if self.is_prefix:
            return self.sequence.prefix_compositions[self.index - 1]
        return (
            self.sequence.composition
            - self.sequence.prefix_compositions[self.cleavage_index]
        )


class Cleavage(NamedTuple):
    prefix: Fragment
    suffix: Fragment
