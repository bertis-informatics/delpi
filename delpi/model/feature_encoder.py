import torch
import torch.nn as nn

from delpi.model.pos_encoder import PositionalEncoder1D, PositionalEncoder2D
from delpi.model.input import ExpPeakInput, TheoPeakInput


PE_2D_DIM = 24  # 2*2*log2(64)
PE_1D_DIM = 8  # 2*log2(16)

THEO_IS_PRECURSOR_IDX = TheoPeakInput.IS_PRECURSOR.index - len(TheoPeakInput)
EXP_IS_PRECURSOR_IDX = ExpPeakInput.IS_PRECURSOR.index - len(ExpPeakInput)


class TheoPeakEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 192,
        max_seq_len: int = 64,
    ):
        super().__init__()

        self.pe_2d = PositionalEncoder2D(
            max_len_dim1=max_seq_len, max_len_dim2=max_seq_len, embed_size=PE_2D_DIM
        )
        concat_input_dim = (input_dim - 2) + PE_2D_DIM
        self.proj = nn.Linear(concat_input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # PE for (cleavage_index, rev_cleavage_index)
        pe_embed = self.pe_2d(x[..., -2], x[..., -1])
        precursor_peak_mask = x[..., THEO_IS_PRECURSOR_IDX] > 0
        pe_embed[precursor_peak_mask, :] = 0
        x_cat = torch.cat([x[..., :-2], pe_embed], dim=-1)
        x_out = self.proj(x_cat)

        return x_out


class ExpPeakEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 192,
        max_seq_len: int = 64,
        max_time_len: int = 12,
    ):
        super().__init__()

        self.pe_2d = PositionalEncoder2D(
            max_len_dim1=max_seq_len, max_len_dim2=max_seq_len, embed_size=PE_2D_DIM
        )
        self.pe_1d = PositionalEncoder1D(max_len=max_time_len, embed_size=PE_1D_DIM)

        concat_input_dim = (input_dim - 3) + PE_2D_DIM + PE_1D_DIM
        self.proj = nn.Linear(concat_input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PE for (cleavage_index, rev_cleavage_index)
        pe_embed = self.pe_2d(x[..., -2], x[..., -1])
        precursor_peak_mask = x[..., EXP_IS_PRECURSOR_IDX] > 0
        pe_embed[precursor_peak_mask, :] = 0

        # PE for time_index
        time_embed = self.pe_1d(x[..., -3])
        x_cat = torch.cat([x[..., :-3], time_embed, pe_embed], dim=-1)
        x_out = self.proj(x_cat)

        return x_out
