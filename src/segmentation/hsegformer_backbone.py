
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import SegformerConfig, SegformerModel


class HSegFormerBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = timm.create_model(
            "resnet50",
            pretrained=False,
            features_only=True,
            out_indices=(1, 2, 3, 4),
            in_chans=1,
        )

        config = SegformerConfig(
            num_channels=1,
            hidden_sizes=[64, 128, 320, 512],
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            num_attention_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            output_hidden_states=True,
        )

        self.segformer = SegformerModel(config)

        self.resnet_projections = nn.ModuleList([
            nn.Conv2d(256, 64, 1),
            nn.Conv2d(512, 128, 1),
            nn.Conv2d(1024, 256, 1),
            nn.Conv2d(2048, 512, 1),
        ])

        self.segformer_projections = nn.ModuleList([
            nn.Conv2d(64, 64, 1),
            nn.Conv2d(128, 128, 1),
            nn.Conv2d(320, 256, 1),
            nn.Conv2d(512, 512, 1),
        ])

    def forward(self, x):
        cnn_features = self.resnet(x)

        expected_resnet = [
            (256, 64, 64),
            (512, 32, 32),
            (1024, 16, 16),
            (2048, 8, 8),
        ]

        for i, feature in enumerate(cnn_features):
            expected = expected_resnet[i]
            actual = tuple(feature.shape[1:])
            if actual != expected:
                raise RuntimeError(
                    f"Unexpected ResNet stage {i + 1}: "
                    f"expected {expected}, got {actual}"
                )

        transformer_features = self.segformer(
            pixel_values=x
        ).hidden_states

        expected_transformer = [
            (64, 64, 64),
            (128, 32, 32),
            (320, 16, 16),
            (512, 8, 8),
        ]

        for i, feature in enumerate(transformer_features):
            expected = expected_transformer[i]
            actual = tuple(feature.shape[1:])
            if actual != expected:
                raise RuntimeError(
                    f"Unexpected SegFormer stage {i + 1}: "
                    f"expected {expected}, got {actual}"
                )

        fused_features = []

        for i in range(4):
            cnn = self.resnet_projections[i](
                cnn_features[i]
            )

            transformer = self.segformer_projections[i](
                transformer_features[i]
            )

            if cnn.shape[-2:] != transformer.shape[-2:]:
                transformer = F.interpolate(
                    transformer,
                    size=cnn.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            fused_features.append(
                torch.cat([cnn, transformer], dim=1)
            )

        return fused_features
