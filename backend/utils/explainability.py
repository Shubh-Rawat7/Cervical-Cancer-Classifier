"""Explainability tools for the Herlev Mamba classifier."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

CLASS_NAMES = ["Normal", "CIN1", "CIN2", "CIN3", "Cancer"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.detach().clone().cpu()
    if t.dim() == 4:
        t = t.squeeze(0)
    for channel in range(3):
        t[channel] = t[channel] * IMAGENET_STD[channel] + IMAGENET_MEAN[channel]
    t = t.permute(1, 2, 0).clamp(0, 1).numpy()
    return (t * 255).astype(np.uint8)


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, image: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        device = next(self.model.parameters()).device
        image = image.to(device)
        self.model.zero_grad(set_to_none=True)
        logits, _, feature_maps = self.model(image, return_features=True, retain_feature_maps=True)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = logits[:, target_class].sum()
        score.backward()

        target_map = feature_maps[-1]
        gradients = target_map.grad
        if gradients is None:
            raise RuntimeError("Grad-CAM gradients were not captured")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * target_map).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-9
        return cam

    def visualize(self, image: torch.Tensor, heatmap: np.ndarray, true_label: int | None = None, pred_label: int | None = None, save_path: str | None = None):
        img_np = denormalize(image)
        cmap = cm.get_cmap("jet")
        colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
        overlay = (0.6 * img_np + 0.4 * colored).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        title_parts = []
        if true_label is not None:
            title_parts.append(f"True: {CLASS_NAMES[true_label]}")
        if pred_label is not None:
            title_parts.append(f"Pred: {CLASS_NAMES[pred_label]}")
        if title_parts:
            fig.suptitle(" | ".join(title_parts))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=160, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


class FeatureMapVisualizer:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, image: torch.Tensor) -> List[np.ndarray]:
        device = next(self.model.parameters()).device
        image = image.to(device)
        with torch.no_grad():
            logits, _, feature_maps = self.model(image, return_features=True, retain_feature_maps=False)
        maps = []
        for feature_map in feature_maps:
            fmap = feature_map.mean(dim=1, keepdim=True)
            fmap = F.interpolate(fmap, size=image.shape[-2:], mode="bilinear", align_corners=False)
            fmap = fmap.squeeze().detach().cpu().numpy()
            fmap -= fmap.min()
            fmap /= fmap.max() + 1e-9
            maps.append(fmap)
        return maps

    def visualize(self, image: torch.Tensor, maps: List[np.ndarray], save_path: str | None = None):
        img_np = denormalize(image)
        cols = len(maps) + 1
        fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
        axes[0].imshow(img_np)
        axes[0].set_title("Input")
        axes[0].axis("off")
        for index, fmap in enumerate(maps, start=1):
            axes[index].imshow(fmap, cmap="magma")
            axes[index].set_title(f"Scale {index}")
            axes[index].axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=160, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


class AttentionMapVisualizer:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, image: torch.Tensor) -> Tuple[np.ndarray, List[np.ndarray]]:
        device = next(self.model.parameters()).device
        image = image.to(device)
        with torch.no_grad():
            logits, _, feature_maps, attention = self.model(image, return_features=True, return_attention=True)

        if attention is None:
            raise RuntimeError("Attention weights were not returned by the model")

        cls_attention = attention[:, 0, 1:]
        heatmaps: List[np.ndarray] = []
        start = 0
        for grid_size in self.model.grid_sizes:
            token_count = grid_size * grid_size
            scale_attention = cls_attention[:, start : start + token_count].mean(dim=1, keepdim=True)
            scale_attention = scale_attention.view(-1, 1, grid_size, grid_size)
            scale_attention = F.interpolate(scale_attention, size=image.shape[-2:], mode="bilinear", align_corners=False)
            scale_attention = scale_attention.squeeze().detach().cpu().numpy()
            scale_attention -= scale_attention.min()
            scale_attention /= scale_attention.max() + 1e-9
            heatmaps.append(scale_attention)
            start += token_count

        combined = np.mean(heatmaps, axis=0)
        return combined, heatmaps

    def visualize(self, image: torch.Tensor, heatmap: np.ndarray, save_path: str | None = None):
        img_np = denormalize(image)
        colored = (cm.get_cmap("viridis")(heatmap)[:, :, :3] * 255).astype(np.uint8)
        overlay = (0.6 * img_np + 0.4 * colored).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np)
        axes[0].set_title("Input")
        axes[0].axis("off")
        axes[1].imshow(heatmap, cmap="viridis")
        axes[1].set_title("Attention Map")
        axes[1].axis("off")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=160, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def explain_batch(model, loader, device, n_samples: int = 8, output_dir: str = "explainability_output"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gcam = GradCAM(model)
    fmap_vis = FeatureMapVisualizer(model)
    attn_vis = AttentionMapVisualizer(model)

    rows = min(n_samples, len(loader.dataset))
    fig, axes = plt.subplots(rows, 4, figsize=(16, rows * 3))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    samples_shown = 0
    for batch in loader:
        images = batch["image"]
        labels = batch["label"]
        for index in range(images.size(0)):
            if samples_shown >= rows:
                break
            image = images[index : index + 1].to(device)
            label = int(labels[index].item())
            with torch.no_grad():
                logits = model(image)
                prediction = int(logits.argmax(dim=1).item())

            cam = gcam(image, target_class=prediction)
            fmap = fmap_vis(image)[-1]
            attn, _ = attn_vis(image)
            overlay = (0.6 * denormalize(image) + 0.4 * (cm.get_cmap("viridis")(attn)[:, :, :3] * 255).astype(np.uint8)).astype(np.uint8)

            axes[samples_shown, 0].imshow(denormalize(image))
            axes[samples_shown, 0].axis("off")
            axes[samples_shown, 0].set_ylabel(f"True: {CLASS_NAMES[label]}\nPred: {CLASS_NAMES[prediction]}", rotation=0, labelpad=90, va="center")
            axes[samples_shown, 1].imshow(cam, cmap="jet")
            axes[samples_shown, 1].axis("off")
            axes[samples_shown, 2].imshow(fmap, cmap="magma")
            axes[samples_shown, 2].axis("off")
            axes[samples_shown, 3].imshow(overlay)
            axes[samples_shown, 3].axis("off")
            samples_shown += 1
        if samples_shown >= rows:
            break

    for axis, title in zip(axes[0], ["Input", "Grad-CAM", "Feature Map", "Attention"]):
        axis.set_title(title)

    plt.tight_layout()
    out_file = output_path / "explainability_grid.png"
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close()
    return out_file
