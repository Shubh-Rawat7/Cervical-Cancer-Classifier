"""Ensemble inference utilities for Herlev Mamba checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import CLASS_NAMES, DEFAULT_IMAGE_SIZE
from dataset import build_eval_transform, build_tta_transforms
from models.model import get_class_names, load_model


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device):
    return load_model(checkpoint_path, device=device)


def predict_image_with_tta(model, image: Image.Image, image_size: int, device: torch.device, tta_views: int = 5) -> np.ndarray:
    transforms = build_tta_transforms(image_size=image_size)[: max(1, int(tta_views))]
    probs = []
    with torch.no_grad():
        for transform in transforms:
            tensor = transform(image=np.asarray(image))["image"].unsqueeze(0).to(device)
            logits = model(tensor)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
    return np.mean(probs, axis=0)


def ensemble_predict(models, loader: DataLoader, device: torch.device, tta_views: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_probs = []
    all_targets = []
    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1].cpu().numpy()
        batch_probs = []
        for model in models:
            with torch.no_grad():
                logits = model(images)
                if tta_views > 1:
                    probs = []
                    for image in images:
                        probs.append(predict_image_with_tta(model, Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)), image_size=images.shape[-1], device=device, tta_views=tta_views))
                    batch_probs.append(np.stack(probs, axis=0))
                else:
                    batch_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        avg_probs = np.mean(batch_probs, axis=0)
        all_probs.append(avg_probs)
        all_targets.append(labels)
    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    preds = probs.argmax(axis=1)
    return probs, preds, targets


def print_metrics(preds: np.ndarray, targets: np.ndarray, probs: np.ndarray) -> None:
    print(f"Ensemble accuracy: {(preds == targets).mean():.4f}")
    print(classification_report(targets, preds, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(targets, preds))
    try:
        auc = roc_auc_score(np.eye(len(CLASS_NAMES))[targets], probs, multi_class="ovr", average="macro")
        print(f"ROC-AUC (macro OvR): {auc:.4f}")
    except Exception as exc:
        print(f"ROC-AUC unavailable: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Herlev ensemble inference")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tta-views", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    models = [load_checkpoint_model(checkpoint, device=device) for checkpoint in args.checkpoints]
    class_names = get_class_names(args.checkpoints[0])

    dataset = ImageFolder(args.data_dir, transform=build_eval_transform(image_size=args.image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    probs, preds, targets = ensemble_predict(models, loader, device=device, tta_views=args.tta_views)
    print(f"Class names: {class_names}")
    print_metrics(preds, targets, probs)


if __name__ == "__main__":
    main()
