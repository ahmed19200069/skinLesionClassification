"""
MobileNetV2-based classifier for skin lesion classification.
I replaced the original ImageNet head with a custom two-layer FC head
and added helpers to freeze/unfreeze backbone layers for the different strategies.

Three strategies this supports:
  feature_extraction  - backbone stays frozen, only the head gets trained
  progressive         - start frozen, then unfreeze backbone in stages
  full_finetune       - train everything end-to-end from the beginning
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


NUM_CLASSES = 7


def _build_head(in_features: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )


class SkinLesionMobileNetV2(nn.Module):
    """MobileNetV2 with a custom classification head for 7-class skin lesion task."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights)

        # keep the feature extractor, replace the classifier
        self.features = backbone.features          # 19 sequential blocks
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        in_feat       = backbone.classifier[1].in_features   # 1280
        self.head     = _build_head(in_feat, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

    def freeze_backbone(self):
        """Freeze all backbone params - used for strategy 1 and start of strategy 2."""
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int):
        """
        Unfreeze the last n blocks in self.features.
        MobileNetV2 has 19 blocks (indices 0-18).
        Example schedule for progressive unfreezing:
          epoch 0-9  -> freeze all
          epoch 10   -> unfreeze last 5  (blocks 14-18)
          epoch 15   -> unfreeze last 10 (blocks 9-18)
        """
        total = len(self.features)
        start = max(0, total - n)
        for i, block in enumerate(self.features):
            requires = i >= start
            for p in block.parameters():
                p.requires_grad = requires

    def unfreeze_all(self):
        """Unfreeze everything - used for full_finetune strategy."""
        for p in self.parameters():
            p.requires_grad = True

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def frozen_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(strategy: str, num_classes: int = NUM_CLASSES) -> SkinLesionMobileNetV2:
    """Build a model configured for the given strategy."""
    model = SkinLesionMobileNetV2(num_classes=num_classes, pretrained=True)

    if strategy == "feature_extraction":
        model.freeze_backbone()

    elif strategy == "progressive":
        # start frozen; train.py handles the progressive unfreezing schedule
        model.freeze_backbone()

    elif strategy == "full_finetune":
        model.unfreeze_all()

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return model


def model_info(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb   = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    return {"total_params": total, "trainable_params": trainable, "size_mb": round(size_mb, 2)}


if __name__ == "__main__":
    for strat in ("feature_extraction", "progressive", "full_finetune"):
        m = build_model(strat)
        info = model_info(m)
        print(f"{strat:20s} | trainable={info['trainable_params']:,} "
              f"| total={info['total_params']:,} | {info['size_mb']} MB")
        x = torch.randn(2, 3, 224, 224)
        print(f"  output shape: {m(x).shape}")
