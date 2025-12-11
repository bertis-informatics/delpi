import torch
from torch import nn

from timm.models.vision_transformer import Block


class Permute(nn.Module):
    """Simple module for tensor permutation"""

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class BasicBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):

    def __init__(
        self, in_channels, out_channels, conv1_kernel_size=7, layers=[1, 1, 1]
    ):
        super(ResNet1D, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv1d(
            in_channels,
            self.in_channels,
            kernel_size=conv1_kernel_size,
            stride=1,
            padding="same",
            # padding = (conv1_kernel_size - 1) // 2
        )

        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(BasicBlock1D, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1D, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(BasicBlock1D, out_channels, layers[2], stride=1)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class BiLSTM(nn.Module):

    def __init__(self, embedding_dim=128, num_layers=1, return_sequences=False):
        super().__init__()

        self.hidden_dim = embedding_dim
        self.return_sequences = return_sequences

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Fully connected layer to map from num_layers*hidden_dim*2 to output_dim
        # num_directions = 2 if self.lstm.bidirectional else 1
        # final_hidden_dim = num_layers * num_directions * embedding_dim
        # self.fc = nn.Linear(final_hidden_dim, embedding_dim)

        # Initialize the initial hidden and cell states as learnable parameters
        self.h0 = nn.Parameter(torch.zeros(2 * num_layers, 1, embedding_dim))
        self.c0 = nn.Parameter(torch.zeros(2 * num_layers, 1, embedding_dim))

    def forward(self, x_emb):

        batch_size = x_emb.shape[0]

        # Initialize the initial hidden and cell states for each batch
        h0 = self.h0.repeat(1, batch_size, 1)
        c0 = self.c0.repeat(1, batch_size, 1)

        # Pass through LSTM
        output, (hn, cn) = self.lstm(x_emb, (h0, c0))

        if self.return_sequences:
            # Return all sequence outputs for MS2 prediction
            # output shape: (batch_size, seq_len, 2 * hidden_dim)
            return output
        else:
            # Return only final hidden state for RT prediction
            # Reshape hn to concatenate the forward and backward hidden states
            hidden = hn.permute(1, 0, 2).reshape(batch_size, -1)
            return hidden


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        return_sequences: bool = False,
    ):
        super().__init__()
        self.return_sequences = return_sequences

        # Only add CLS token for RT prediction (when return_sequences=False)
        if not return_sequences:
            self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
            torch.nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

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

    def forward(self, x_emb):

        batch_size = x_emb.shape[0]

        if self.return_sequences:
            # For MS2 prediction: return all sequence outputs
            x = self.transformer(x_emb)
            x = self.layer_norm(x)
            return x
        else:
            # For RT prediction: use CLS token and return single output
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x_emb), dim=1)
            x = self.transformer(x)
            x = self.layer_norm(x)
            return x[:, 0]  # Return only CLS token output
