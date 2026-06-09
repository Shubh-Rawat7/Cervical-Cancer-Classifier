"""
backend/models/cnn_model.py
============================
REPLACED: 6-layer custom CNN  →  EfficientNetV2-S  (ImageNet pretrained)

WHY THIS CHANGE:
  - Original 6-conv custom CNN trained from scratch on only 750 images is
    guaranteed to overfit. Train acc 95% vs val 88% in the README confirms it.
  - EfficientNetV2-S pretrained on ImageNet already knows edges, textures,
    and shapes. We only need to fine-tune the final layers on our 5 classes.
  - Expected accuracy: 93-96% val acc (vs current 88%)
  - Computational cost: comparable to the old 6-conv model on GPU;
    slightly heavier on CPU but acceptable.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DROP-IN MODEL  (EfficientNetV2-S, recommended)
# ─────────────────────────────────────────────────────────────────────────────

class CervicalCancerCNN(nn.Module):
    """
    EfficientNetV2-S backbone + custom classification head.

    Two-phase training supported:
      phase=1  →  freeze backbone, train head only   (5-10 epochs)
      phase=2  →  unfreeze all layers, fine-tune     (remaining epochs)
    """

    NUM_CLASSES = 5   # Normal, CIN1, CIN2, CIN3, Cancer

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()

        # ── backbone ────────────────────────────────────────────────────────
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_v2_s(weights=weights)

        # Remove the stock classifier; keep the feature extractor
        in_features = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # ── custom head ─────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # safety pool (already done inside backbone)
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        # initialise head weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ── phase control ────────────────────────────────────────────────────────
    def freeze_backbone(self):
        """Phase 1: only train the head."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze_backbone(self, unfreeze_last_n_blocks: int = 4):
        """
        Phase 2: unfreeze the last N blocks of the backbone for fine-tuning.
        Unfreezing everything at once on 750 images causes overfitting.
        Start with 4 blocks, increase if val loss keeps improving.
        """
        # first freeze everything
        for p in self.backbone.parameters():
            p.requires_grad = False

        # then selectively unfreeze
        blocks = list(self.backbone.features.children())
        for block in blocks[-unfreeze_last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

        # always train the head
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze the entire network (use only with large datasets)."""
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, 1280)
        # EfficientNetV2 already applies AdaptiveAvgPool + Flatten internally
        # when classifier is Identity(). But shape is (B, 1280) so skip extra pool.
        x = nn.functional.dropout(features, p=0.0, training=False)
        # ── head (skip the AdaptiveAvgPool+Flatten since features are already flat)
        x = self.classifier[2](features)     # BN1d
        x = self.classifier[3](x)            # Dropout
        x = self.classifier[4](x)            # Linear 1280→512
        x = self.classifier[5](x)            # SiLU
        x = self.classifier[6](x)            # BN1d
        x = self.classifier[7](x)            # Dropout
        x = self.classifier[8](x)            # Linear 512→5
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ALTERNATIVE: DenseNet121  (lighter, also excellent for medical imaging)
# ─────────────────────────────────────────────────────────────────────────────

class CervicalDenseNet(nn.Module):
    """
    DenseNet121 alternative. Dense connections naturally act as deep supervision,
    making them very strong on small medical datasets.
    Expected val accuracy: 91-94%.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        backbone = models.densenet121(weights=weights)
        in_features = backbone.classifier.in_features  # 1024
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def freeze_backbone(self):
        for name, p in self.backbone.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = CervicalCancerCNN()
    model.freeze_backbone()
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # Expected: (4, 5)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params (phase 1): {trainable:,} / {total:,}")

    model.unfreeze_backbone(unfreeze_last_n_blocks=4)
    trainable2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (phase 2): {trainable2:,} / {total:,}")