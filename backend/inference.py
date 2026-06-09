"""Herlev cervical cell inference with TTA and checkpoint ensembling."""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image

from config import CLASS_NAMES, DEFAULT_IMAGE_SIZE, MODEL_PATH
from dataset import build_tta_transforms
from models.model import get_class_names, load_model


def _apply_transform(transform, image: Image.Image) -> torch.Tensor:
    transformed = transform(image=np.asarray(image))
    return transformed["image"] if isinstance(transformed, dict) else transformed


def temperature_scale(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    return logits / temperature


def predict_with_tta(
    model,
    image: Image.Image,
    image_size: int,
    tta_views: int,
    device: torch.device,
    temperature: float = 1.0,
) -> np.ndarray:
    transforms = build_tta_transforms(image_size)
    transforms = transforms[: max(1, int(tta_views))]
    probs = []
    model.eval()
    with torch.no_grad():
        for transform in transforms:
            tensor = _apply_transform(transform, image).unsqueeze(0).to(device)
            logits = temperature_scale(model(tensor), temperature=temperature)
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy()[0])
    return np.mean(probs, axis=0)


def resolve_model_paths(
    model_paths: Sequence[str | Path],
    model_dir: str | Path | None = None,
    model_glob: str = "fold_*_best.pt",
) -> List[Path]:
    resolved_paths = [Path(p) for p in model_paths if p]
    if resolved_paths:
        return resolved_paths

    if model_dir:
        root = Path(model_dir)
        if root.is_dir():
            resolved_paths = sorted(root.glob(model_glob))
            if resolved_paths:
                return [Path(p) for p in resolved_paths]

    return [Path(MODEL_PATH)]


def predict_image(
    image_path: str | Path,
    model_paths: Sequence[str | Path],
    image_size: int = DEFAULT_IMAGE_SIZE,
    tta_views: int = 5,
    temperature: float = 1.0,
    device: str | torch.device | None = None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image = Image.open(image_path).convert("RGB")
    ensemble_probs = []
    class_names = list(CLASS_NAMES)

    for model_path in model_paths:
        model = load_model(model_path, device=device)
        class_names = get_class_names(model_path)
        ensemble_probs.append(
            predict_with_tta(model, image, image_size=image_size, tta_views=tta_views, device=device, temperature=temperature)
        )

    probs = np.mean(ensemble_probs, axis=0)
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "class_names": list(class_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Herlev inference with TTA and ensemble averaging")
    parser.add_argument("--model-path", nargs="*", default=[], help="Paths to model checkpoint files")
    parser.add_argument("--model-dir", type=str, default="", help="Directory containing model checkpoints to ensemble")
    parser.add_argument("--model-glob", type=str, default="fold_*_best.pt", help="Pattern to match checkpoint files in model-dir")
    parser.add_argument("--image-path", required=True, help="Input image file or directory")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--tta-views", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    model_paths = resolve_model_paths(args.model_path, args.model_dir, args.model_glob)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    if image_path.is_dir():
        results = []
        image_files = sorted([p for p in image_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}])
        for file_path in image_files:
            result = predict_image(file_path, model_paths, image_size=args.image_size, tta_views=args.tta_views, temperature=args.temperature, device=device)
            result["image"] = file_path.name
            results.append(result)
            print(f"{file_path.name}: {result['predicted_class']} ({result['confidence']:.4f})")
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        result = predict_image(image_path, model_paths, image_size=args.image_size, tta_views=args.tta_views, temperature=args.temperature, device=device)
        print(json.dumps(result, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
