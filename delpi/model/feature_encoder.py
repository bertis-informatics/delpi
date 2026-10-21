import torch
import torch.nn as nn
import torch.nn.functional as F

from delpi.constants import COMMON_NEUTRAL_LOSSES
from delpi.model.pos_encoder import PositionalEncoder1D, PositionalEncoder2D
from delpi.model.input import (
    ExpPeakInput,
    TheoPeakInput,
    ContextInput,
    IonType,
    AcquisitionMode,
)

PE_2D_DIM = 24
PE_1D_DIM = 8


def valid_tokens(x, valid_mask=None):
    if valid_mask is None:
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    if valid_mask.shape != x.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError(
            "valid_mask must be a boolean tensor matching token dimensions"
        )
    return valid_mask


def categorical_features(x, ion_index, loss_index):
    continuous = [i for i in range(x.shape[-1]) if i not in (ion_index, loss_index)]
    return torch.cat(
        (
            x[..., continuous],
            F.one_hot(x[..., ion_index].long(), len(IonType)).to(x.dtype),
            F.one_hot(x[..., loss_index].long(), len(COMMON_NEUTRAL_LOSSES)).to(
                x.dtype
            ),
        ),
        dim=-1,
    )


class TheoPeakEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=192, max_seq_len=64):
        super().__init__()
        self.pe_2d = PositionalEncoder2D(
            max_len_dim1=max_seq_len, max_len_dim2=max_seq_len, embed_size=PE_2D_DIM
        )
        self.proj = nn.Linear(
            input_dim - 4 + len(IonType) + len(COMMON_NEUTRAL_LOSSES) + PE_2D_DIM,
            embed_dim,
        )

    def forward(self, x, valid_mask=None):
        valid = valid_tokens(x, valid_mask)
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        pe = self.pe_2d(x[..., -2], x[..., -1])
        positional = valid & (
            x[..., TheoPeakInput.ION_TYPE.index] != int(IonType.PRECURSOR)
        )
        pe = pe.masked_fill(~positional.unsqueeze(-1), 0)
        features = categorical_features(
            x[..., :-2], TheoPeakInput.ION_TYPE.index, TheoPeakInput.NEUTRAL_LOSS.index
        )
        return self.proj(torch.cat((features, pe), dim=-1)).masked_fill(
            ~valid.unsqueeze(-1), 0
        )


class ExpPeakEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=192, max_seq_len=64, max_time_len=12):
        super().__init__()
        # The MAE decoder passes metadata only (the last input_dim fields).
        offset = len(ExpPeakInput) - input_dim
        self.ion_index = ExpPeakInput.ION_TYPE.index - offset
        self.loss_index = ExpPeakInput.NEUTRAL_LOSS.index - offset
        self.pe_2d = PositionalEncoder2D(
            max_len_dim1=max_seq_len, max_len_dim2=max_seq_len, embed_size=PE_2D_DIM
        )
        self.pe_1d = PositionalEncoder1D(max_len=max_time_len, embed_size=PE_1D_DIM)
        self.proj = nn.Linear(
            input_dim
            - 5
            + len(IonType)
            + len(COMMON_NEUTRAL_LOSSES)
            + PE_2D_DIM
            + PE_1D_DIM,
            embed_dim,
        )

    def forward(self, x, valid_mask=None):
        valid = valid_tokens(x, valid_mask)
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        pe = self.pe_2d(x[..., -2], x[..., -1])
        pe = pe.masked_fill(
            (~valid | (x[..., self.ion_index] == int(IonType.PRECURSOR))).unsqueeze(-1),
            0,
        )
        time = self.pe_1d(x[..., -3]).masked_fill(~valid.unsqueeze(-1), 0)
        features = categorical_features(x[..., :-3], self.ion_index, self.loss_index)
        return self.proj(torch.cat((features, time, pe), dim=-1)).masked_fill(
            ~valid.unsqueeze(-1), 0
        )


class ContextEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Linear(len(ContextInput) - 1 + len(AcquisitionMode), embed_dim)

    def forward(self, x):
        if x.ndim != 2 or x.shape[-1] != len(ContextInput):
            raise ValueError("context must have shape (batch, len(ContextInput))")
        mode = F.one_hot(
            x[..., ContextInput.ACQUISITION_MODE.index].long(), len(AcquisitionMode)
        ).to(x.dtype)
        return self.proj(torch.cat((mode, x[..., 1:]), dim=-1)).unsqueeze(1)


def token_attention_mask(x, theo_valid_mask, exp_valid_mask):
    valid = torch.cat(
        (
            torch.ones((x.shape[0], 2), dtype=torch.bool, device=x.device),
            theo_valid_mask,
            exp_valid_mask,
        ),
        dim=1,
    )
    bias = torch.zeros((x.shape[0], 1, 1, x.shape[1]), dtype=x.dtype, device=x.device)
    return valid, bias.masked_fill(~valid[:, None, None, :], float("-inf"))
