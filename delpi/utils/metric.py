import math

import torch
from torchmetrics import Metric, PearsonCorrCoef
from torchmetrics.utilities import dim_zero_cat
import torch.nn.functional as F


class RecallAtFDR(Metric):

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, fdr_cutoff=0.01, **kwargs):
        super().__init__(**kwargs)

        self.fdr_cutoff = fdr_cutoff
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds, target):
        self.preds.append(preds.detach().to("cpu", non_blocking=True).flatten())
        self.target.append(target.detach().to("cpu", non_blocking=True).flatten())

    def compute(self):

        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        ii = torch.argsort(preds, descending=True)

        target = target[ii]
        tgt_cum = target.cumsum(0)
        dec_cum = torch.arange(1, len(target) + 1, device=target.device) - tgt_cum
        fdr_hat = dec_cum / tgt_cum.clamp(min=1)

        passed = fdr_hat <= self.fdr_cutoff
        if not passed.any():
            return torch.tensor(0, device=target.device)

        k = passed.nonzero(as_tuple=False)[-1, 0]

        # return tgt_cum[k]
        num_targets = tgt_cum[-1]

        return tgt_cum[k] / num_targets


class SafePearsonCorrCoef(PearsonCorrCoef):
    """Accumulate Pearson in float64 for sparse, low-intensity fragments.

    Both inputs and states use float64 to avoid the float32 near-zero variance
    cutoff. An undefined correlation (constant inputs or too few observations)
    is reported as 0 by convention, not as a measured correlation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_dtype(torch.float64)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.numel() == 0:
            return
        super().update(preds.to(torch.float64), target.to(torch.float64))

    def compute(self):
        return torch.nan_to_num(super().compute(), nan=0.0, posinf=0.0, neginf=0.0)


class SpectralAngle(Metric):
    """
    Spectral Angle Similarity metric in [0, 1].
    1.0 = identical spectra, 0.0 = orthogonal.

    With ignore_empty_targets=True, average only spectra with observed peaks.
    A zero prediction for a nonzero target still contributes a score of 0.
    """

    def __init__(self, dist_sync_on_step=False, ignore_empty_targets=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_empty_targets = ignore_empty_targets
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target: [..., D]
        if self.ignore_empty_targets:
            present = target.ne(0).any(dim=-1)
            preds, target = preds[present], target[present]
        cos_sim = F.cosine_similarity(preds, target, dim=-1, eps=1e-8)
        # Robustness: treat negative cosine (angle > 90°) as 0 similarity
        cos_sim = cos_sim.clamp(min=0.0, max=1.0)
        angle = torch.acos(cos_sim)  # [0, π/2]
        score = 1.0 - angle / (math.pi / 2)  # [0, 1]
        self.sum += score.sum()
        self.total += score.numel()

    def compute(self):
        # No observed spectra means no defined SA; report 0 by convention.
        return self.sum / self.total.clamp_min(1)
