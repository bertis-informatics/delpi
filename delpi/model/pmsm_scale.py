import torch

from delpi.model.input import TheoPeakInput, ExpPeakInput


class PeptideMultiSpectraMatchScaler(torch.nn.Module):

    def __init__(
        self,
        mz_denom: float = 1000,
        charge_denom: float = 5,
        isotope_denom: float = 5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mz_denom = mz_denom
        # self.time_denom = time_denom
        # self.seq_len_denom = seq_len_denom
        self.charge_denom = charge_denom
        self.isotope_denom = isotope_denom

    def forward(self, x_theo, x_exp):

        ## theoretical peaks
        x_theo[..., TheoPeakInput.MZ.index] /= self.mz_denom
        x_theo[..., TheoPeakInput.CHARGE.index] /= self.charge_denom
        x_theo[..., TheoPeakInput.ISOTOPE_INDEX.index] /= self.isotope_denom

        # experimental peaks
        x_exp[..., ExpPeakInput.MS_LEVEL.index] -= 1.0  # 0 for MS1, 1 for MS2
        x_exp[..., ExpPeakInput.CHARGE.index] /= self.charge_denom
        x_exp[..., ExpPeakInput.ISOTOPE_INDEX.index] /= self.isotope_denom

        return x_theo, x_exp
