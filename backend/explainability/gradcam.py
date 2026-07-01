"""Grad-CAM utilities for the MambaVision backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def _target_layer(model):
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise AttributeError("Model does not expose a backbone for Grad-CAM")
    blocks = getattr(backbone, "blocks", None)
    if blocks:
        return list(blocks)[-1]
    children = list(backbone.children())
    if not children:
        raise AttributeError("Backbone does not expose any target layers")
    return children[-1]


def save_gradcam_visualization(model, image_path: str | Path, transform, output_path: str | Path, device) -> Path | None:
    """Generate and save a Grad-CAM overlay for a single image."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        import cv2
        import torch
    except Exception:
        return None

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image=np.asarray(image))["image"].unsqueeze(0).to(device)
    cam = GradCAM(model=model, target_layers=[_target_layer(model)])
    grayscale_cam = cam(input_tensor=tensor)[0]

    resized = np.asarray(image.resize((tensor.shape[-1], tensor.shape[-2]))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(resized, grayscale_cam, use_rgb=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return output_path