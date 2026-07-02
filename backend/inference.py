"""
inference.py — TTA ensemble inference + Grad-CAM for the API.

Usage:
    from inference import CervicalInference
    inf = CervicalInference("checkpoints/best_model.pth")
    result = inf.predict("path/to/image.jpg")
    # result = {"class": "CIN2", "confidence": 0.87, "probabilities": {...}}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.cnn_model import build_model
from utils.transforms import tta_transforms, val_transform


def _extract_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "ema_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise TypeError("Checkpoint does not contain a compatible state_dict")

CLASS_NAMES = ["Normal", "CIN1", "CIN2", "CIN3", "Cancer"]


class CervicalInference:
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        num_classes: int = 5,
        use_tta: bool = True,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_tta = use_tta
        self.transforms = tta_transforms() if use_tta else [val_transform()]
        self.model = build_model(num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = _extract_state_dict(checkpoint)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        Returns:
            {
                "class": "CIN2",
                "confidence": 0.87,
                "probabilities": {"Normal": 0.02, "CIN1": 0.05, ...},
                "risk_level": "High"
            }
        """
        img = Image.open(image_path).convert("RGB")

        # TTA: average softmax probabilities across all augmentations
        all_probs = []
        for tfm in self.transforms:
            x = tfm(img).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        pred_idx = int(avg_probs.argmax())
        confidence = float(avg_probs[pred_idx])

        risk_map = {
            "Normal": "None",
            "CIN1":   "Low",
            "CIN2":   "Moderate",
            "CIN3":   "High",
            "Cancer": "Critical",
        }
        pred_class = CLASS_NAMES[pred_idx]

        return {
            "class":         pred_class,
            "confidence":    round(confidence, 4),
            "probabilities": {c: round(float(p), 4)
                              for c, p in zip(CLASS_NAMES, avg_probs)},
            "risk_level":    risk_map[pred_class],
        }

    def predict_batch(self, image_paths: list[str]) -> list[dict]:
        return [self.predict(p) for p in image_paths]


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM (optional, for explainability in the frontend)
# ──────────────────────────────────────────────────────────────────────────────

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import cv2

    class CervicalGradCAM:
        def __init__(self, inference: CervicalInference):
            self.inference = inference
            target_layer = inference.model.backbone.blocks[-1]
            self.cam = GradCAM(
                model=inference.model,
                target_layers=[target_layer],
            )

        def generate(self, image_path: str, output_path: str) -> str:
            """
            Generates a Grad-CAM heatmap overlay and saves to output_path.
            Returns output_path.
            """
            img = Image.open(image_path).convert("RGB")
            tfm = val_transform()
            x = tfm(img).unsqueeze(0)

            grayscale_cam = self.cam(input_tensor=x)[0]

            img_resized = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
            overlay = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

            cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            return output_path

except ImportError:
    CervicalGradCAM = None  # type: ignore


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python inference.py <checkpoint.pth> <image.jpg>")
        sys.exit(1)
    inf = CervicalInference(sys.argv[1])
    result = inf.predict(sys.argv[2])
    for k, v in result.items():
        print(f"  {k}: {v}")