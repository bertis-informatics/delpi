import torch
from timm.models.vision_transformer import Block

from delpi.model.feature_encoder import TheoPeakEncoder, ExpPeakEncoder


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

    def forward(self, x_theo, x_exp, mask_ratio):

        x_exp, mask, backward_indexes = self.random_masking(x_exp, mask_ratio)
        x = self._forward_impl(x_theo, x_exp)

        return x, mask, backward_indexes

    def _forward_impl(self, x_theo: torch.Tensor, x_exp: torch.Tensor):

        # x_exp = self.exp_peak_encoder(x_exp[:, :, :-2]) + self.pos_encoder(
        #     x_exp[:, :, -2], x_exp[:, :, -1]
        # )
        # x_theo = self.theo_peak_encoder(x_theo[:, :, :-1]) + self.pos_encoder_1d(
        #     x_theo[:, :, -1]
        # )
        x_theo = self.theo_peak_encoder(x_theo)
        x_exp = self.exp_peak_encoder(x_exp)

        # append cls_token and theoretical_peak_tokens,
        #       [B, len_keep, D] -> [B, len_keep+1+len_theo, D]
        # cls_token = self.cls_token + self.pos_encoder.pos_embed[0, :]
        cls_tokens = self.cls_token.expand(x_theo.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x_theo, x_exp), dim=1)

        x = self.transformer(x)
        x = self.layer_norm(x)

        return x


def test_encoder():
    from delpi.model.input import THEORETICAL_PEAK, EXPERIMENTAL_PEAK

    encoder = Encoder(
        theo_peak_dim=len(THEORETICAL_PEAK), exp_peak_dim=len(EXPERIMENTAL_PEAK)
    )

    x_theo = torch.rand(32, 51, len(THEORETICAL_PEAK))

    x_exp = torch.rand(32, 40, len(EXPERIMENTAL_PEAK))

    # assign cleavage indices
    x_exp[:, :, -2] = torch.randint(0, 30, x_exp.shape[:2])

    # assign scan indices
    x_exp[:, :, -1] = torch.randint(0, 9, x_exp.shape[:2])

    # assign cleavage indices
    x_theo[:, :, -1] = torch.randint(0, 30, x_theo.shape[:2])

    self = encoder
