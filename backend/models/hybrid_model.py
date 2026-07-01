"""Hybrid MambaVision + handcrafted-feature classifier for Herlev cervical cells."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from config import CLASS_NAMES, DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES

try:
    import timm
except Exception as exc:  # pragma: no cover - optional dependency
    timm = None
    _TIMM_IMPORT_ERROR = exc
else:
    _TIMM_IMPORT_ERROR = None

try:
    from features.feature_extractor import FEATURE_NAMES, extract_medical_features
except Exception:  # pragma: no cover - allow the model module to import without extractor
    FEATURE_NAMES = [f"feature_{index}" for index in range(28)]
    extract_medical_features = None


HANDCRAFTED_DIM = 28
FUSION_DIM = 1308
BACKBONE_DIM = 1280


def _resolve_backbone_name(preferred: str) -> str:
    if timm is None:
        raise ImportError("timm is required for the hybrid Herlev classifier") from _TIMM_IMPORT_ERROR

    preferred = (preferred or "mambavision_small").strip()
    candidates = [
        preferred,
        "mambavision_small",
        "mambavision_base",
        "mambaout_small_rw.sw_in12k_ft_in1k",
        "convnextv2_tiny",
        "tf_efficientnetv2_s",
    ]

    available = set(timm.list_models())
    for candidate in candidates:
        if candidate in available:
            return candidate
        matches = timm.list_models(f"*{candidate}*")
        if matches:
            return matches[0]

    raise ValueError(f"Could not resolve backbone '{preferred}' in timm")


def _strip_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    stripped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "ema_model.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        stripped[new_key] = value
    return stripped


def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "ema_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return _strip_prefix(value)
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return _strip_prefix(checkpoint)
    raise ValueError("Checkpoint does not contain a compatible state dict")


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(features, ord=2))
    if norm <= 1e-12:
        return features.astype(np.float32)
    return (features / norm).astype(np.float32)


class HerlevHybridClassifier(nn.Module):
    """MambaVision-Small + handcrafted-feature fusion classifier.

    Architecture:
        1280 backbone features + 28 handcrafted features = 1308 fused features
        1308 -> Linear(512) -> BatchNorm -> ReLU -> Dropout(0.4)
             -> Linear(256) -> BatchNorm -> ReLU -> Dropout(0.4)
             -> Linear(5)

    The forward pass returns logits by default so existing training code keeps
    working with cross-entropy-based losses. Use `return_probs=True` to obtain
    softmax probabilities.
    """

    def __init__(
        self,
        backbone: str = "mambavision_small",
        num_classes: int = NUM_CLASSES,
        image_size: int = DEFAULT_IMAGE_SIZE,
        dropout: float = 0.4,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = _resolve_backbone_name(backbone)
        self.num_classes = num_classes
        self.image_size = image_size
        self.dropout_rate = dropout
        self.handcrafted_dim = HANDCRAFTED_DIM
        self.fusion_dim = FUSION_DIM
        self.backbone_dim = BACKBONE_DIM

        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        resolved_backbone_dim = int(getattr(self.backbone, "num_features", BACKBONE_DIM))
        self.backbone_projector = (
            nn.Identity() if resolved_backbone_dim == BACKBONE_DIM else nn.Linear(resolved_backbone_dim, BACKBONE_DIM)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if isinstance(self.backbone_projector, nn.Linear):
            for parameter in self.backbone_projector.parameters():
                parameter.requires_grad = False

    def freeze_classifier_only(self) -> None:
        """Freeze the backbone and train only the classifier head."""
        self._freeze_backbone()
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def freeze_backbone(self) -> None:
        """Alias for freezing feature extraction layers."""
        self._freeze_backbone()

    def unfreeze_last_blocks(self, num_blocks: int = 2) -> None:
        """Unfreeze the last `num_blocks` transformer blocks of the backbone."""
        self._freeze_backbone()
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is not None:
            block_list = list(blocks)
            for block in block_list[-num_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        else:
            children = list(self.backbone.children())
            for child in children[-num_blocks:]:
                for parameter in child.parameters():
                    parameter.requires_grad = True

        if isinstance(self.backbone_projector, nn.Linear):
            for parameter in self.backbone_projector.parameters():
                parameter.requires_grad = True

        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self, unfreeze_last_n_blocks: int = 2) -> None:
        """Progressively unfreeze the last MambaVision blocks."""
        self.unfreeze_last_blocks(num_blocks=unfreeze_last_n_blocks)

    def unfreeze_all(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
        if isinstance(self.backbone_projector, nn.Linear):
            for parameter in self.backbone_projector.parameters():
                parameter.requires_grad = True

    def _extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.ndim == 4:
            features = torch.nn.functional.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        features = self.backbone_projector(features)
        if features.ndim != 2:
            features = features.flatten(1)
        return features

    def _extract_handcrafted_features(self, x: torch.Tensor) -> torch.Tensor:
        if extract_medical_features is None:
            return torch.zeros((x.size(0), HANDCRAFTED_DIM), device=x.device, dtype=x.dtype)

        mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        images = x.detach().float() * std + mean
        images = torch.clamp(images, 0.0, 1.0)
        images = (images * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        features = []
        for image in images:
            vector = extract_medical_features(image)
            features.append(vector)

        handcrafted = torch.tensor(np.asarray(features, dtype=np.float32), device=x.device, dtype=x.dtype)
        if handcrafted.ndim == 1:
            handcrafted = handcrafted.unsqueeze(0)
        handcrafted = handcrafted / torch.clamp(handcrafted.norm(p=2, dim=1, keepdim=True), min=1e-12)
        return handcrafted

    def forward(
        self,
        x: torch.Tensor,
        handcrafted_features: Optional[torch.Tensor] = None,
        return_probs: bool = False,
    ) -> torch.Tensor:
        backbone_features = self._extract_backbone_features(x)

        if handcrafted_features is None:
            handcrafted_features = self._extract_handcrafted_features(x)
        else:
            handcrafted_features = handcrafted_features.to(device=x.device, dtype=backbone_features.dtype)
            if handcrafted_features.ndim == 1:
                handcrafted_features = handcrafted_features.unsqueeze(0)
            handcrafted_features = handcrafted_features / torch.clamp(
                handcrafted_features.norm(p=2, dim=1, keepdim=True), min=1e-12
            )

        fused = torch.cat([backbone_features, handcrafted_features], dim=1)
        logits = self.classifier(fused)
        if return_probs:
            return self.softmax(logits)
        return logits


def build_model(
    backbone: str = "mambavision_small",
    num_classes: int = NUM_CLASSES,
    image_size: int = DEFAULT_IMAGE_SIZE,
    dropout: float = 0.4,
    pretrained: bool = True,
) -> HerlevHybridClassifier:
    return HerlevHybridClassifier(
        backbone=backbone,
        num_classes=num_classes,
        image_size=image_size,
        dropout=dropout,
        pretrained=pretrained,
    )


def load_model(checkpoint_path: str | Path, device: Optional[torch.device | str] = None) -> HerlevHybridClassifier:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)

    metadata = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    checkpoint_backbone = checkpoint.get("backbone", "mambavision_small") if isinstance(checkpoint, dict) else "mambavision_small"
    checkpoint_num_classes = checkpoint.get("num_classes", NUM_CLASSES) if isinstance(checkpoint, dict) else NUM_CLASSES
    checkpoint_image_size = checkpoint.get("image_size", DEFAULT_IMAGE_SIZE) if isinstance(checkpoint, dict) else DEFAULT_IMAGE_SIZE
    checkpoint_dropout = checkpoint.get("dropout", 0.4) if isinstance(checkpoint, dict) else 0.4

    backbone = metadata.get("backbone", checkpoint_backbone)
    num_classes = int(metadata.get("num_classes", checkpoint_num_classes))
    image_size = int(metadata.get("image_size", checkpoint_image_size))
    dropout = float(metadata.get("dropout", checkpoint_dropout))

    model = build_model(
        backbone=backbone,
        num_classes=num_classes,
        image_size=image_size,
        dropout=dropout,
        pretrained=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_class_names(checkpoint_path: str | Path | None = None) -> list[str]:
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "class_names" in checkpoint:
                return list(checkpoint["class_names"])
            config = checkpoint.get("config", {})
            if "class_names" in config:
                return list(config["class_names"])
    return list(CLASS_NAMES)