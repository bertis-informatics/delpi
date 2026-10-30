import torch
import torch.nn as nn

from delpi.model.pos_encoder import PositionalEncoder1D, PositionalEncoder2D
from delpi.model.input import (
    ExpPeakInput,
    TheoPeakInput,
    ContextInput,
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


class TheoPeakEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=192, max_seq_len=64):
        super().__init__()
        self.pe_2d = PositionalEncoder2D(
            max_len_dim1=max_seq_len, max_len_dim2=max_seq_len, embed_size=PE_2D_DIM
        )
        self.proj = nn.Linear(input_dim - 2 + PE_2D_DIM, embed_dim)

    def forward(self, x, valid_mask=None):
        valid = valid_tokens(x, valid_mask)
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        pe = self.pe_2d(x[..., -2], x[..., -1])
        positional = valid & (x[..., TheoPeakInput.IS_PRECURSOR.index] == 0)
        pe = pe.masked_fill(~positional.unsqueeze(-1), 0)
        return self.proj(torch.cat((x[..., :-2], pe), dim=-1)).masked_fill(
            ~valid.unsqueeze(-1), 0
        )


class ExpPeakEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=192, max_seq_len=64, max_time_len=12):
        super().__init__()
        # The MAE decoder passes metadata only (the last input_dim fields).
        offset = len(ExpPeakInput) - input_dim
        self.precursor_index = ExpPeakInput.IS_PRECURSOR.index - offset
        self.pe_2d = PositionalEncoder2D(
            max_len_dim1=max_seq_len, max_len_dim2=max_seq_len, embed_size=PE_2D_DIM
        )
        self.pe_1d = PositionalEncoder1D(max_len=max_time_len, embed_size=PE_1D_DIM)
        self.proj = nn.Linear(
            input_dim - 3 + PE_2D_DIM + PE_1D_DIM,
            embed_dim,
        )

    def forward(self, x, valid_mask=None):
        valid = valid_tokens(x, valid_mask)
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        pe = self.pe_2d(x[..., -2], x[..., -1])
        pe = pe.masked_fill(
            (~valid | (x[..., self.precursor_index] != 0)).unsqueeze(-1),
            0,
        )
        time = self.pe_1d(x[..., -3]).masked_fill(~valid.unsqueeze(-1), 0)
        return self.proj(torch.cat((x[..., :-3], time, pe), dim=-1)).masked_fill(
            ~valid.unsqueeze(-1), 0
        )


class ContextEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Linear(len(ContextInput), embed_dim)

    def forward(self, x):
        if x.ndim != 2 or x.shape[-1] != len(ContextInput):
            raise ValueError("context must have shape (batch, len(ContextInput))")
        return self.proj(x).unsqueeze(1)


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
