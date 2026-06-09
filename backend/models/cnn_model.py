"""
cnn_model.py — EfficientNetV2-S backbone with SE channel attention head.
Drop-in replacement for the original 6-layer CNN.
Targets 5-class cervical cancer classification: Normal, CIN1, CIN2, CIN3, Cancer.
"""

import torch
import torch.nn as nn
import timm


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class CervicalClassifier(nn.Module):
    """
    EfficientNetV2-S pretrained on ImageNet → SE attention → 5-class head.

    Two-phase training:
        Phase 1 (freeze=True)  : only head + SE block trained
        Phase 2 (freeze=False) : full fine-tune with lower LR on backbone
    """

    NUM_CLASSES = 5
    BACKBONE_OUT = 1280  # EfficientNetV2-S final feature dim

    def __init__(self, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()
        self.num_classes = num_classes

        # Pretrained backbone — features only (no classifier)
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=True,
            num_classes=0,   # remove timm head
            global_pool="",  # remove global pool so we get spatial features
        )
        feat_dim = self.backbone.num_features  # 1280 for v2-s

        self.se = SEBlock(feat_dim, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        self._freeze_backbone()

    # ------------------------------------------------------------------
    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, unfreeze_last_n_blocks: int = 3):
        """Progressively unfreeze the last N blocks of the backbone."""
        blocks = list(self.backbone.blocks)
        for block in blocks[-unfreeze_last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        # Always keep BN stats frozen on shallow layers
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    def forward(self, x):
        feat = self.backbone.forward_features(x)  # (B, C, H, W)
        feat = self.se(feat)
        feat = self.pool(feat).flatten(1)          # (B, C)
        return self.head(feat)


def build_model(num_classes: int = 5, dropout: float = 0.4) -> CervicalClassifier:
    return CervicalClassifier(num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    m = build_model()
    x = torch.randn(2, 3, 224, 224)
    print(m(x).shape)  # should be (2, 5)