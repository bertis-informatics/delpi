import torch
import torch.nn as nn

from delpi.model.mae import Encoder


class DelPiModel(nn.Module):
    def __init__(self, encoder: Encoder, classifier: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.transform = None

    def forward(self, x_theo: torch.Tensor, x_exp: torch.Tensor):

        if self.transform is not None:
            x_theo, x_exp = self.transform(x_theo, x_exp)

        x = self.encoder._forward_impl(x_theo, x_exp)
        x_feature = x[:, 0]
        logits = self.classifier(x_feature)
        return logits, x_feature
