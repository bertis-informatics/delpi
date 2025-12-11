import math
import torch
import torch.nn as nn


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0

    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concat([emb_sin, emb_cos], dim=1)  # (M, D)

    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.concat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, max_len_dim1, max_len_dim2):

    grid_h = torch.arange(max_len_dim1, dtype=torch.float32)
    grid_w = torch.arange(max_len_dim2, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, max_len_dim1, max_len_dim2])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    # if cls_token:
    #     pos_embed = torch.concat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed


class PositionalEncoder2D(nn.Module):

    def __init__(self, embed_size, max_len_dim1, max_len_dim2):

        super().__init__()

        self.max_len_dim1 = max_len_dim1
        self.max_len_dim2 = max_len_dim2
        pos_embed = get_2d_sincos_pos_embed(embed_size, max_len_dim1, max_len_dim2)
        self.register_buffer("pos_embed", pos_embed)

    def forward(self, dim1_indices, dim2_indices):

        pos_embed = self.pos_embed
        embed_size = pos_embed.shape[-1]

        pos_embed_indices = self.max_len_dim2 * dim1_indices + dim2_indices

        pos_embed_indices = pos_embed_indices.to(torch.int64)

        pos_embed = torch.gather(
            pos_embed[None, :, :].expand(dim1_indices.shape[0], -1, -1),
            dim=1,
            index=pos_embed_indices.unsqueeze(-1).repeat(1, 1, embed_size),
        )

        return pos_embed


class PositionalEncoder1D(nn.Module):

    def __init__(self, embed_size, max_len):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size)
        )

        pos_embed = torch.zeros(max_len, embed_size)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_embed", pos_embed)

    def forward(self, cleavage_indices):

        pos_embed = self.pos_embed
        embed_size = pos_embed.shape[-1]

        pos_embed_indices = cleavage_indices.to(torch.int64)
        pos_embed = torch.gather(
            pos_embed[None, :, :].expand(cleavage_indices.shape[0], -1, -1),
            dim=1,
            index=pos_embed_indices.unsqueeze(-1).repeat(1, 1, embed_size),
        )

        return pos_embed


class PositionalEncoding(torch.nn.Module):
    def __init__(self, embed_size, max_len=128):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-math.log(max_len) / embed_size)
        )

        pos_embed = torch.zeros(max_len, embed_size)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_embed", pos_embed)

    def forward(self, x):
        return x + self.pos_embed[None, : x.size(1), :]


def test_1d():
    pos_encoder = PositionalEncoder1D(embed_size=128, max_len=50)

    batch_size = 2
    cleavage_indices = torch.zeros([batch_size, 45])
    cleavage_indices[0, :15] = torch.arange(15)
    cleavage_indices[0, 15:30] = torch.arange(15)
    cleavage_indices[0, 30:45] = torch.arange(15)

    pos_embed = pos_encoder(cleavage_indices)
    pos_embed[0, 15, :]
    pos_embed[0, 14, :]


def test():
    import time

    batch_size = 32
    # assign cleavage indices
    # x[:, :, cleavage_index_dim] = torch.randint(0, max_seq_len, x.shape[:2])

    cleavage_indices = torch.zeros([batch_size, 45])
    scan_indices = torch.zeros([batch_size, 45])

    cleavage_indices[0, :15] = torch.arange(15)
    scan_indices[0, :15] = torch.arange(15)

    cleavage_indices[0, 15:30] = torch.arange(15)
    scan_indices[0, 15:30] = torch.arange(15)

    cleavage_indices[0, 30:45] = torch.arange(15)
    scan_indices[0, 30:45] = torch.arange(15)

    # assign scan indices
    # x[:, :, scan_index_dim] = torch.randint(0, max_seq_len, x.shape[:2])
    embed_size = 256
    max_seq_len = 32
    max_scan_len = 32

    peak_pos_encoder = PositionalEncoder2D(embed_size, max_seq_len, max_scan_len)

    pe = peak_pos_encoder(cleavage_indices, scan_indices)

    from matplotlib import pyplot as plt

    plt.figure()
    plt.pcolor(pe[0, :, :])
    plt.savefig("./temp/pos_enc.jpg")

    # pe[0, :15, 16:] - pe[0, 15:30, 16:]
