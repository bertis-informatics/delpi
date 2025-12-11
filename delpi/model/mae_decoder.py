import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

from delpi.model.feature_encoder import ExpPeakEncoder


class Decoder(torch.nn.Module):
    def __init__(
        self,
        theo_peak_dim,
        exp_peak_dim,
        encoder_embed_dim,
        output_dim=3,
        embed_dim=128,
        depth=8,
        num_heads=16,
        qkv_bias=True,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.input_encoder = nn.Linear(encoder_embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # self.input_meta_encoder = nn.Linear(exp_peak_dim - 2 - output_dim, embed_dim)
        # self.pos_encoder = PositionalEncoder2D(embed_dim, max_seq_len=32, max_scan_len=9)

        self.input_meta_encoder = ExpPeakEncoder(
            input_dim=exp_peak_dim - output_dim,
            embed_dim=embed_dim,
        )

        self.transformer = torch.nn.Sequential(
            *[Block(embed_dim, num_heads, qkv_bias=qkv_bias) for _ in range(depth)]
        )
        self.layer_norm = torch.nn.LayerNorm(embed_dim, eps=1e-6)
        self.predictor = nn.Linear(embed_dim, output_dim)

        torch.nn.init.normal_(self.mask_token, std=0.02)
        # self.apply(self._init_weights)

    def forward(self, x_in, x_latent, backward_indexes, n_theo_tokens):

        n_extra_tokens = n_theo_tokens + 1  # +1 for cls_token

        x = self.input_encoder(x_latent)

        # append mask tokens to sequence
        # [B, len_masked, D']
        n_masked_tokens = x_in.shape[1] - (x_latent.shape[1] - n_extra_tokens)
        mask_tokens = self.mask_token.repeat(x.shape[0], n_masked_tokens, 1)

        # [B, len_keep, D'], [B, len_masked, D'] --> [B, L, D']
        x_ = torch.cat(
            [x[:, n_extra_tokens:, :], mask_tokens], dim=1
        )  # remove global feature
        x_ = torch.gather(
            x_, dim=1, index=backward_indexes.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        # x_ = x_ + self.input_meta_encoder(x_in[:, :, self.output_dim : -2])
        # x_ = x_ + self.pos_encoder(x_in[:, :, -2], x_in[:, :, -1])
        x_ += self.input_meta_encoder(x_in[..., self.output_dim :])

        # append theo_peak_tokens & cls token
        x = torch.cat([x[:, :n_extra_tokens, :], x_], dim=1)

        x = self.layer_norm(self.transformer(x))

        # predictor projection
        x = self.predictor(x)

        # remove theo_peak_tokens & cls token
        x = x[:, n_extra_tokens:, :]

        return x
