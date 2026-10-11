
import torch
import torch.nn as nn
import torch.nn.functional as F


class NABlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.norm = nn.GroupNorm(
            num_groups=min(32, channels),
            num_channels=channels,
        )

        self.gelu = nn.GELU()

        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xa = self.dwconv(x)
        xa = self.pointwise(xa)
        xa = self.norm(xa)
        xa = self.gelu(xa)

        pooled = F.adaptive_avg_pool2d(xa, 1)
        pooled = pooled.flatten(1)

        attention = self.fc1(pooled)
        attention = self.gelu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        attention = attention.view(
            x.shape[0],
            x.shape[1],
            1,
            1,
        )

        return attention * xa


class HSegFormerFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.nablocks = nn.ModuleList([
            NABlock(128),
            NABlock(256),
            NABlock(512),
            NABlock(1024),
        ])

        self.compression = nn.ModuleList([
            nn.Conv2d(128, 64, 1),
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(1024, 512, 1),
        ])

    def forward(self, features):
        if len(features) != 4:
            raise ValueError("HSegFormer fusion expects four feature stages")

        outputs = []

        for i, feature in enumerate(features):
            feature = self.nablocks[i](feature)
            feature = self.compression[i](feature)
            outputs.append(feature)

        return outputs
