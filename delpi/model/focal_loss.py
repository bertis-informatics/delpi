import torch
import torch.nn.functional as F
import torch.nn as nn


## reference: https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf


class AsymmetricFocalLoss(nn.Module):
    def __init__(
        self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8, reduction="mean"
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets, return_score=False):

        targets = targets.float()

        y_proba = torch.sigmoid(logits)
        p_pos = y_proba
        p_neg = 1.0 - y_proba

        # Positive term
        loss_pos = (
            torch.pow(p_neg, self.gamma_pos)
            * torch.log(torch.clamp(p_pos, min=self.eps))
        ) * targets

        # Negative term
        if self.clip > 0.0:
            p_pos = torch.clamp(p_pos - self.clip, min=0.0)
        loss_neg = (
            torch.pow(p_pos, self.gamma_neg)
            * torch.log(torch.clamp(p_neg, min=self.eps))
        ) * (1.0 - targets)

        loss = -(loss_pos + loss_neg)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        if return_score:
            return loss, y_proba
        return loss


class FocalLoss(nn.Module):
    def __init__(
        self, gamma: float = 2.0, alpha: float = 0.25, eps=1e-8, reduction="mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):

        y_true = targets.float()
        y_proba = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, y_true, reduction="none")

        if self.gamma > 1:
            p_t = y_proba * y_true + (1 - y_proba) * (1 - y_true)
            loss = ce_loss * ((1 - p_t) ** self.gamma)
        else:
            loss = ce_loss

        if self.alpha > 0:
            alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# def focal_loss(
#     logits: torch.Tensor,
#     y_true: torch.Tensor,
#     gamma: float = 2.0,
#     alpha: float = 0.25,
#     eps: float = 1e-8,
# ) -> torch.Tensor:

#     # https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
#     y_true = y_true.to(torch.float32)
#     y_proba = logits.sigmoid()
#     ce_loss = F.binary_cross_entropy_with_logits(logits, y_true, reduction="none")

#     if gamma > 1:
#         p_t = y_proba * y_true + (1 - y_proba) * (1 - y_true)
#         loss = ce_loss * ((1 - p_t) ** gamma)
#     else:
#         loss = ce_loss

#     if alpha > 0:
#         alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
#         loss = alpha_t * loss

#     return loss.mean(), y_proba
