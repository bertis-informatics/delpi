import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _safe_mean(t: Tensor) -> Tensor:
    return t.mean() if t.numel() else t.new_tensor(0.0)


# PU risk
#   https://arxiv.org/abs/1703.00593?utm_source=chatgpt.com
# PNU risk
#   https://arxiv.org/pdf/1803.04663
#   https://proceedings.mlr.press/v97/hsieh19c/hsieh19c.pdf


class PNURiskLoss(nn.Module):

    def __init__(
        self,
        pi_p: float,
        eta: float = -0.3,
    ):
        """_summary_
        Args:
            pi_p (float): 전체 모집단에서 positive 비율( class prior, θ ).
            eta (float, optional): _description_. Defaults to -0.3.
                • 0   : PN risk
                • 0-1 : PN과 PU convex combination (unlabeled가 negative 에 가깝다고 볼 때 )
                • -1-0 → PN과 NU convex combination(unlabeled가 **positive 에 가깝다**고 볼 때 )
        Raises:
            ValueError: _description_
        """
        super().__init__()
        if not -1.0 <= eta <= 1.0:
            raise ValueError("eta must be in [-1, 1]")
        self.pi_p = float(pi_p)
        self.eta = float(eta)

    def forward(self, z, p_mask, n_mask, u_mask):
        # ------ PN risk --------------------------------------------------
        rp = _safe_mean(F.softplus(-z[p_mask]))  # label 1
        rn = _safe_mean(F.softplus(z[n_mask]))  # label 0
        r_pn = self.pi_p * rp + (1 - self.pi_p) * rn

        # ------ PU risk --------------------------------------------------
        r_bar_p = -_safe_mean(z[p_mask])  # E_P[ℓ₊ − ℓ₋] = −E_P[z]
        r_u_neg = _safe_mean(F.softplus(z[u_mask]))  # label 0
        r_pu = self.pi_p * r_bar_p + r_u_neg

        # ------ NU risk --------------------------------------------------
        r_bar_n = _safe_mean(z[n_mask])  #  E_N[ℓ₋ − ℓ₊] =  E_N[z]
        r_u_pos = _safe_mean(F.softplus(-z[u_mask]))  # label 1
        r_nu = (1 - self.pi_p) * r_bar_n + r_u_pos

        # ------ η‑mix ----------------------------------------------------
        risk = (
            ((1 - self.eta) * r_pn + self.eta * r_pu)
            if self.eta >= 0
            else ((1 + self.eta) * r_pn - self.eta * r_nu)
        )

        return torch.clamp(risk, min=0.0)

    def __forward(
        self,
        logits: Tensor,
        p_mask: Tensor,
        n_mask: Tensor,
        u_mask: Tensor,
    ) -> Tensor:

        base_loss = nn.BCEWithLogitsLoss(reduction="none")

        ones = torch.ones_like(logits)
        zeros = torch.zeros_like(logits)

        # ---------- PN risk ---------------------------------------------- #
        rp = _safe_mean(base_loss(logits[p_mask], ones[p_mask]))
        rn = _safe_mean(base_loss(logits[n_mask], zeros[n_mask]))
        r_pn = self.pi_p * rp + (1.0 - self.pi_p) * rn

        # ---------- PU risk  (unlabeled 를 negative 로 본다) -------------- #
        # composite loss  l̄ = l(+1) − l(−1)
        l_pos_p = base_loss(logits[p_mask], ones[p_mask])
        l_neg_p = base_loss(logits[p_mask], zeros[p_mask])
        r_bar_p = _safe_mean(l_pos_p - l_neg_p)  # E_P[ l̄ ]

        r_u_neg = _safe_mean(base_loss(logits[u_mask], zeros[u_mask]))
        r_pu = self.pi_p * r_bar_p + r_u_neg

        # ---------- NU risk  (unlabeled 를 positive 로 본다) -------------- #
        l_pos_u = _safe_mean(base_loss(logits[u_mask], ones[u_mask]))
        l_neg_n = base_loss(logits[n_mask], zeros[n_mask])
        l_pos_n = base_loss(logits[n_mask], ones[n_mask])
        r_bar_n = _safe_mean(l_neg_n - l_pos_n)  # E_N[ l̄(−·) ]
        r_nu = (1.0 - self.pi_p) * r_bar_n + l_pos_u

        # ---------- 최종 PNU risk ---------------------------------------- #
        if self.eta >= 0:  # unlabeled 가 negative 성향
            risk = (1.0 - self.eta) * r_pn + self.eta * r_pu
        else:  # unlabeled 가 **positive 성향**
            risk = (1.0 + self.eta) * r_pn - self.eta * r_nu

        # Non-negative clip, Kiryo et al.(2017)
        risk = torch.clamp(risk, min=0.0)

        return risk


class NURiskLoss(nn.Module):
    """Non‑negative NU risk with (optional) Asymmetric Focal Loss."""

    def __init__(
        self,
        pi_p: float = 0.10,
        base_loss: nn.Module = nn.BCEWithLogitsLoss(reduction="none"),
        # use_afl: bool = True,
        # gamma_pos: float = 0.0,
        # gamma_neg: float = 2.0,
    ):
        super().__init__()
        self.pi_p = pi_p
        self.base_loss = base_loss
        # self.base = (
        #     AsymmetricFocalLoss(gamma_pos, gamma_neg, reduction="none")
        #     if use_afl
        #     else nn.BCEWithLogitsLoss(reduction="none")
        # )

    def forward(self, logits, targets, return_score=False):

        mask_neg = targets == 0
        mask_u = ~mask_neg

        loss = self._forward_impl(logits, mask_neg, mask_u)

        if return_score:
            return loss, torch.sigmoid(logits)

        return loss

    def _forward_impl(self, logits, mask_neg, mask_u):
        """
        logits  : Tensor[B]
        mask_neg: BoolTensor[B]  -- reliable negatives
        mask_u  : BoolTensor[B]  -- unlabeled (noisy "positives")
        """
        y_pos = torch.ones_like(logits)
        y_neg = torch.zeros_like(logits)

        # term 1: (1‑pi_p) * E_N[ℓ(g,0)]
        Rn = _safe_mean(self.base_loss(logits[mask_neg], y_neg[mask_neg]))

        # helper for E_N[ℓ(g,1)]  (negatives fed as if y=1)
        Rn_pos = _safe_mean(self.base_loss(logits[mask_neg], y_pos[mask_neg]))

        # term 2: E_U[ℓ(g,1)] - (1‑pi_p) * E_N[ℓ(g,1)]
        Ru = _safe_mean(self.base_loss(logits[mask_u], y_pos[mask_u]))
        unbiased = Ru - (1 - self.pi_p) * Rn_pos

        loss = (1 - self.pi_p) * Rn + torch.clamp(unbiased, min=0.0)
        return loss


def test():

    l1 = PNURiskLoss(pi_p=0.5)

    logit = torch.rand(512)
    pnu_idx = torch.randint(2, size=(logit.shape[0],))
    p_mask = pnu_idx == 1
    n_mask = pnu_idx == 0
    u_mask = pnu_idx == 2

    l1(logit, p_mask, n_mask, u_mask)

    # z, p_mask, n_mask, u_mask)
