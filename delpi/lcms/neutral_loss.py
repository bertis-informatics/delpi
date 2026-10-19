import polars as pl

from delpi.chem.composition import Composition
from delpi.constants import COMMON_NEUTRAL_LOSSES

# Physical Enum codes follow the shared fragment-intensity loss indices.
NEUTRAL_LOSS_DTYPE = pl.Enum(COMMON_NEUTRAL_LOSSES)


class NeutralLoss:

    NO_LOSS = None
    H2O = None
    NH3 = None
    H3O4P = None
    H4COS = None
    neutral_loss_map = {}

    def __init__(self, name, symbol, composition) -> None:
        self.name = name
        self.symbol = symbol
        self.composition = composition

    def __hash__(self) -> int:
        return hash(self.symbol)

    def __eq__(self, other):
        if isinstance(other, NeutralLoss):
            return self.symbol == other.symbol
        return False

    @classmethod
    def get_common_neutral_losses(cls):
        return [cls.NO_LOSS, cls.H2O, cls.NH3]

    @classmethod
    def get(cls, symbol):
        """Return a statically defined neutral loss by its DelPi symbol."""
        return cls.neutral_loss_map[symbol]


NeutralLoss.NO_LOSS = NeutralLoss("", "NoLoss", Composition.ZERO)
NeutralLoss.H2O = NeutralLoss("-H2O", "H2O", Composition.H2O)
NeutralLoss.NH3 = NeutralLoss("-NH3", "NH3", Composition.NH3)

# modification specific neutral-losses (mod-loss)
NeutralLoss.H3O4P = NeutralLoss("-H3O4P", "H3O4P", Composition(0, 3, 0, 4, 0, 1))
NeutralLoss.H4COS = NeutralLoss("-H4COS", "H4COS", Composition(1, 4, 0, 1, 1, 0))

NeutralLoss.neutral_loss_map = {
    loss.symbol: loss
    for loss in vars(NeutralLoss).values()
    if isinstance(loss, NeutralLoss)
}
