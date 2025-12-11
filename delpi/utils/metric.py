import math

import torch
from torchmetrics import Metric
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


class SpectralAngle(Metric):
    """
    Spectral Angle Similarity metric in [0, 1].
    1.0 = identical spectra, 0.0 = orthogonal.
    """

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target: [..., D]
        cos_sim = F.cosine_similarity(preds, target, dim=-1, eps=1e-8)
        # Robustness: treat negative cosine (angle > 90°) as 0 similarity
        cos_sim = cos_sim.clamp(min=0.0, max=1.0)
        angle = torch.acos(cos_sim)  # [0, π/2]
        score = 1.0 - angle / (math.pi / 2)  # [0, 1]
        self.sum += score.sum()
        self.total += score.numel()

    def compute(self):
        return self.sum / self.total
