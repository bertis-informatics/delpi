import torch
from timm.models.vision_transformer import Block

from delpi.model.feature_encoder import (
    TheoPeakEncoder,
    ExpPeakEncoder,
    ContextEncoder,
    valid_tokens,
    token_attention_mask,
)


class Encoder(torch.nn.Module):
    def __init__(
        self,
        theo_peak_dim: int,
        exp_peak_dim: int,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 12,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.context_encoder = ContextEncoder(embed_dim)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.theo_peak_encoder = TheoPeakEncoder(
            theo_peak_dim,
            embed_dim=embed_dim,
        )
        self.exp_peak_encoder = ExpPeakEncoder(
            exp_peak_dim,
            embed_dim=embed_dim,
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.transformer = torch.nn.Sequential(
            *[
                Block(embed_dim, num_heads, qkv_bias=qkv_bias, drop_path=dpr[i])
                for i in range(depth)
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(embed_dim, eps=1e-6)

        torch.nn.init.normal_(self.cls_token, std=0.02)

    def random_masking(self, x, mask_ratio):

        B, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        forward_indexes = torch.argsort(torch.rand(B, L, device=x.device), dim=1)
        backward_indexes = torch.argsort(forward_indexes, dim=1)

        x_masked = torch.gather(
            x, dim=1, index=forward_indexes[:, :len_keep].unsqueeze(-1).repeat(1, 1, D)
        )

        # binary mask
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=backward_indexes)

        return x_masked, mask, backward_indexes

    def forward(
        self, x_theo, x_exp, x_context, mask_ratio, theo_valid_mask, exp_valid_mask=None
    ):
        exp_valid = valid_tokens(x_exp, exp_valid_mask)
        x_exp, mask, backward_indexes = self.random_masking(x_exp, mask_ratio)
        mask = mask * exp_valid.to(mask.dtype)
        forward_indexes = torch.argsort(backward_indexes, dim=1)[:, : x_exp.shape[1]]
        exp_valid = torch.gather(exp_valid, 1, forward_indexes)
        x = self._forward_impl(x_theo, x_exp, x_context, theo_valid_mask, exp_valid)
        return x, mask, backward_indexes

    def _forward_impl(
        self, x_theo, x_exp, x_context, theo_valid_mask, exp_valid_mask=None
    ):
        theo_valid = valid_tokens(x_theo, theo_valid_mask)
        exp_valid = valid_tokens(x_exp, exp_valid_mask)
        theo = self.theo_peak_encoder(x_theo, theo_valid)
        exp = self.exp_peak_encoder(x_exp, exp_valid)
        context = self.context_encoder(x_context)
        cls = self.cls_token.expand(x_theo.shape[0], -1, -1)
        x = torch.cat((cls, context, theo, exp), dim=1)
        valid, bias = token_attention_mask(x, theo_valid, exp_valid)
        for block in self.transformer:
            x = block(x, attn_mask=bias).masked_fill(~valid.unsqueeze(-1), 0)
        return self.layer_norm(x).masked_fill(~valid.unsqueeze(-1), 0)
