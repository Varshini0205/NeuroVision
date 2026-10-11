
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hsegformer_fusion import NABlock


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()

        self.gate_projection = nn.Conv2d(
            gate_channels, inter_channels, 1, bias=False
        )
        self.skip_projection = nn.Conv2d(
            skip_channels, inter_channels, 1, bias=False
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

        self.output_projection = nn.Conv2d(
            skip_channels, skip_channels, 1, bias=False
        )

    def forward(self, gate, skip):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(
                gate,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        attention = self.gate_projection(gate)
        attention = attention + self.skip_projection(skip)
        attention = self.relu(attention)
        attention = self.psi(attention)

        return self.output_projection(skip * attention)


class ASPP(nn.Module):
    def __init__(self, channels, rates=(1, 6, 12, 18)):
        super().__init__()

        branches = []

        for rate in rates:
            kernel_size = 3 if rate > 1 else 1
            padding = rate if rate > 1 else 0

            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size,
                        padding=padding,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.GroupNorm(
                        min(32, channels),
                        channels,
                    ),
                    nn.GELU(),
                )
            )

        self.branches = nn.ModuleList(branches)

        self.project = nn.Sequential(
            nn.Conv2d(channels * len(rates), channels, 1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
        )

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        return self.project(torch.cat(outputs, dim=1))


class HSegFormerDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up16 = nn.ConvTranspose2d(
            512, 256, 2, stride=2
        )
        self.gate16 = AttentionGate(256, 256, 128)
        self.refine16 = NABlock(256)
        self.aspp16 = ASPP(256)

        self.up8 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )
        self.gate8 = AttentionGate(128, 128, 64)
        self.refine8 = NABlock(128)

        self.up4 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )
        self.gate4 = AttentionGate(64, 64, 32)
        self.refine4 = NABlock(64)

        self.main_head = nn.Conv2d(64, 1, 1)
        self.aux16 = nn.Conv2d(256, 1, 1)
        self.aux8 = nn.Conv2d(128, 1, 1)

    def forward(self, features):
        if len(features) != 4:
            raise ValueError("Decoder expects four feature stages")

        f4, f8, f16, f32 = features

        x16 = self.up16(f32)
        x16 = x16 + self.gate16(x16, f16)
        x16 = self.refine16(x16)
        x16 = self.aspp16(x16)

        x8 = self.up8(x16)
        x8 = x8 + self.gate8(x8, f8)
        x8 = self.refine8(x8)

        x4 = self.up4(x8)
        x4 = x4 + self.gate4(x4, f4)
        x4 = self.refine4(x4)

        main = self.main_head(x4)
        aux16 = self.aux16(x16)
        aux8 = self.aux8(x8)

        target_size = (256, 256)

        return {
            "main": F.interpolate(
                main,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ),
            "aux16": F.interpolate(
                aux16,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ),
            "aux8": F.interpolate(
                aux8,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ),
        }
