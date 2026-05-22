"""
Week 1 — Baseline CNN built entirely from scratch (no pretrained weights).

Architecture (inspired by VGG-style blocks but kept lightweight for CPU viability):

  Input: (B, 3, 224, 224)

  Block 1: Conv(3→32, 3×3) → BN → ReLU → Conv(32→32, 3×3) → BN → ReLU → MaxPool(2×2) → Drop(0.25)
  Block 2: Conv(32→64, 3×3) → BN → ReLU → Conv(64→64, 3×3) → BN → ReLU → MaxPool(2×2) → Drop(0.25)
  Block 3: Conv(64→128,3×3) → BN → ReLU → Conv(128→128,3×3) → BN → ReLU → MaxPool(2×2) → Drop(0.25)
  Block 4: Conv(128→256,3×3)→ BN → ReLU → Conv(256→256,3×3)→ BN → ReLU → MaxPool(2×2) → Drop(0.25)

  GlobalAvgPool → Flatten
  FC: 256 → 256 → ReLU → Drop(0.5) → 7

Total parameters: ~2.5M  (comparable to MobileNetV2's 2.2M for a fair baseline)
"""

from __future__ import annotations
import torch
import torch.nn as nn


NUM_CLASSES = 7


def _conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _vgg_block(in_ch: int, out_ch: int, dropout: float = 0.25) -> nn.Sequential:
    return nn.Sequential(
        _conv_bn_relu(in_ch,  out_ch),
        _conv_bn_relu(out_ch, out_ch),
        nn.MaxPool2d(2, 2),
        nn.Dropout2d(dropout),
    )


class BaselineCNN(nn.Module):
    """
    Scratch-trained CNN used as the Week-1 baseline.
    Same input size (224×224) as MobileNetV2 for a fair comparison.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_fc: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            _vgg_block(3,    32),   # → (B,  32, 112, 112)
            _vgg_block(32,   64),   # → (B,  64,  56,  56)
            _vgg_block(64,  128),   # → (B, 128,  28,  28)
            _vgg_block(128, 256),   # → (B, 256,  14,  14)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # → (B, 256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def model_info(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb   = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
    return {
        "total_params":     total,
        "trainable_params": trainable,
        "size_mb":          round(size_mb, 2),
    }


if __name__ == "__main__":
    m = BaselineCNN()
    x = torch.randn(2, 3, 224, 224)
    info = model_info(m)
    print(f"Output shape : {m(x).shape}")
    print(f"Parameters   : {info['total_params']:,}")
    print(f"Model size   : {info['size_mb']} MB")
