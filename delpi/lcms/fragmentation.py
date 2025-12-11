import polars as pl

from delpi.lcms.base_ion_type import BaseIonType
from delpi.lcms.neutral_loss import NeutralLoss
from delpi.lcms.data_container import IonTypeContainer
from delpi.constants import C13C12_MASS_DIFF


def get_frag_type_str(
    base_ion_type: BaseIonType, neutral_loss: NeutralLoss, charge: int
):
    if neutral_loss == NeutralLoss.NO_LOSS:
        return f"{base_ion_type.symbol}_z{charge}"
    elif neutral_loss == NeutralLoss.H3O4P:
        return f"{base_ion_type.symbol}_modloss_z{charge}"

    return f"{base_ion_type.symbol}_{neutral_loss.symbol}_z{charge}"


class Fragmentation:

    def __init__(
        self,
        min_charge,
        max_charge,
        max_fragment_isotopes,
        prefix_ion_type=BaseIonType.B,
        suffix_ion_type=BaseIonType.Y,
        neutral_losses=[NeutralLoss.NO_LOSS],
        *args,
        **kwargs,
    ):

        if max_fragment_isotopes > 2:
            raise NotImplementedError()

        self.min_charge = min_charge
        self.max_charge = max_charge
        self.max_fragment_isotopes = max_fragment_isotopes
        self.neutral_losses = neutral_losses
        self.base_ion_types = [prefix_ion_type, suffix_ion_type]
        self.ion_type_df = self._generate_ion_table()

    @property
    def num_ion_types(self):
        return self.ion_type_df.shape[0] // self.max_fragment_isotopes

    def _generate_ion_table(self):
        schema = {
            "base_ion_type": pl.Categorical,
            "neutral_loss": pl.Categorical,
            "charge": pl.UInt8,
            "isotope_index": pl.UInt8,
            "is_prefix": pl.Boolean,
            "offset_mass": pl.Float32,
            "frag_type": pl.String,
        }
        data = list()
        for charge in range(self.min_charge, self.max_charge + 1):
            for neutral_loss in self.neutral_losses:
                for base_ion_type in self.base_ion_types:
                    offset_mass_ = (
                        base_ion_type.offset_composition - neutral_loss.composition
                    ).mass
                    frag_type_str = get_frag_type_str(
                        base_ion_type, neutral_loss, charge
                    )

                    for iso_index in range(self.max_fragment_isotopes):
                        data.append(
                            (
                                base_ion_type.symbol,
                                neutral_loss.symbol,
                                charge,
                                iso_index,
                                base_ion_type.is_prefix,
                                offset_mass_ + C13C12_MASS_DIFF * iso_index,
                                frag_type_str,
                            )
                        )

        return pl.DataFrame(data, schema=schema, orient="row")

    def get_ion_types(self):
        ion_type_df = (
            self.ion_type_df.with_columns(is_modloss=pl.col("neutral_loss") != "NoLoss")
            .sort(pl.col("is_modloss"), ~pl.col("is_prefix"), pl.col("charge"))
            .select(pl.col("is_modloss", "is_prefix", "charge", "offset_mass"))
        )
        return IonTypeContainer(
            ion_type_df["is_prefix"].to_numpy(),
            ion_type_df["charge"].to_numpy(),
            ion_type_df["offset_mass"].to_numpy(),
            ion_type_df["is_modloss"].to_numpy(),
        )
