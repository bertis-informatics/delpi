import numpy as np
from delpi.model.input import ContextInput


def make_context(
    precursor_mz, acquisition_mode, nce, library_nce, isolation_min_mz, isolation_max_mz
):
    """Build unscaled, per-candidate acquisition context for model input."""

    context = np.empty((len(precursor_mz), len(ContextInput)), dtype=np.float32)
    context[:, ContextInput.ACQUISITION_MODE.index] = int(acquisition_mode)
    context[:, ContextInput.NCE.index] = nce
    context[:, ContextInput.LIBRARY_NCE.index] = library_nce
    context[:, ContextInput.ISOLATION_LEFT.index] = precursor_mz - isolation_min_mz
    context[:, ContextInput.ISOLATION_RIGHT.index] = isolation_max_mz - precursor_mz

    return context
