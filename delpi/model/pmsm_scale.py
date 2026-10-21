import torch

from delpi.model.input import TheoPeakInput, ExpPeakInput, ContextInput


class PeptideMultiSpectraMatchScaler(torch.nn.Module):
    """Scale charge/isotopes by 5, NCE by 100, isolation offsets by asinh(x/5).

    Preserve categorical codes, raw MS levels, and caller-owned buffers.
    """

    def __init__(self, charge_denom: float = 5, isotope_denom: float = 5):
        super().__init__()
        self.charge_denom = charge_denom
        self.isotope_denom = isotope_denom

    def forward(self, x_theo, x_exp, x_context):
        x_theo, x_exp, x_context = x_theo.clone(), x_exp.clone(), x_context.clone()
        x_theo[..., TheoPeakInput.CHARGE.index] /= self.charge_denom
        x_theo[..., TheoPeakInput.ISOTOPE_INDEX.index] /= self.isotope_denom
        x_exp[..., ExpPeakInput.CHARGE.index] /= self.charge_denom
        x_exp[..., ExpPeakInput.ISOTOPE_INDEX.index] /= self.isotope_denom
        x_context[..., ContextInput.NCE.index] /= 100.0
        x_context[..., ContextInput.LIBRARY_NCE.index] /= 100.0
        for field in (ContextInput.ISOLATION_LEFT, ContextInput.ISOLATION_RIGHT):
            x_context[..., field.index] = torch.asinh(x_context[..., field.index] / 5.0)
        return x_theo, x_exp, x_context
