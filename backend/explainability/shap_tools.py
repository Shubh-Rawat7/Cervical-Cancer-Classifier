"""SHAP utilities for handcrafted feature explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from features.feature_extractor import FEATURE_NAMES, extract_medical_features


def save_handcrafted_shap_summary(
    model,
    image_array: np.ndarray,
    output_path: str | Path,
    background_samples: Sequence[np.ndarray] | None = None,
) -> Path | None:
    """Save a SHAP summary plot for the handcrafted feature contribution.

    The model's backbone embedding is held fixed while SHAP varies only the
    handcrafted feature vector, which keeps the explanation focused on the
    engineered features.
    """

    try:
        import shap
        import torch
        import matplotlib.pyplot as plt
    except Exception:
        return None

    handcrafted = np.asarray(extract_medical_features(image_array), dtype=np.float32).reshape(1, -1)
    if background_samples:
        background = np.asarray(background_samples, dtype=np.float32)
    else:
        background = handcrafted

    device = next(model.parameters()).device
    image_tensor = torch.zeros((1, 3, getattr(model, "image_size", 224), getattr(model, "image_size", 224)), device=device)
    backbone_features = model._extract_backbone_features(image_tensor).detach().cpu().numpy()[0]

    def predict(handcrafted_batch: np.ndarray) -> np.ndarray:
        batch = torch.tensor(np.asarray(handcrafted_batch, dtype=np.float32), device=device)
        backbone = torch.tensor(np.repeat(backbone_features[None, :], batch.shape[0], axis=0), device=device)
        fused = torch.cat([backbone, batch], dim=1)
        logits = model.classifier(fused)
        return torch.softmax(logits, dim=1).detach().cpu().numpy()

    explainer = shap.KernelExplainer(predict, background)
    shap_values = explainer.shap_values(handcrafted, nsamples=min(128, handcrafted.shape[1] * 8))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, handcrafted, feature_names=FEATURE_NAMES, show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=160)
    plt.close()
    return output_path