"""
backend/utils/dataset_audit.py
================================
Provides code to:
  1. Detect duplicate images (perceptual hash)
  2. Detect corrupted images
  3. Visualize class distribution
  4. Find potentially mislabeled samples (confidence-based outlier detection)

Usage:
    python utils/dataset_audit.py --data-dir ../data/train --output audit_report/
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# Optional: install imagehash for perceptual hashing
try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("Warning: imagehash not installed. pip install ImageHash")

CLASS_NAMES = ["Normal", "CIN1", "CIN2", "CIN3", "Cancer"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Detect corrupted images
# ─────────────────────────────────────────────────────────────────────────────
def find_corrupted_images(data_dir: str) -> list:
    """Return list of (path, error) for unreadable images."""
    corrupted = []
    data_path = Path(data_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    for img_path in data_path.rglob("*"):
        if img_path.suffix.lower() not in image_extensions:
            continue
        try:
            with Image.open(img_path) as img:
                img.verify()          # catches truncated files
            with Image.open(img_path) as img:
                img.load()            # catches lazy-load issues
        except (UnidentifiedImageError, OSError, Exception) as e:
            corrupted.append((str(img_path), str(e)))

    print(f"\n[Corrupted Images] Found {len(corrupted)} corrupted files:")
    for path, err in corrupted:
        print(f"  ✗ {path}: {err}")
    return corrupted


# ─────────────────────────────────────────────────────────────────────────────
# 2. Detect duplicate images
# ─────────────────────────────────────────────────────────────────────────────
def find_duplicates(data_dir: str, hash_size: int = 8) -> dict:
    """
    Detect duplicate images using perceptual hash (pHash).
    Much better than MD5 — catches resized/recompressed duplicates too.
    Returns dict mapping hash → list of paths.
    """
    data_path = Path(data_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    hash_map = defaultdict(list)

    for img_path in data_path.rglob("*"):
        if img_path.suffix.lower() not in image_extensions:
            continue
        try:
            with Image.open(img_path) as img:
                if HAS_IMAGEHASH:
                    h = str(imagehash.phash(img, hash_size=hash_size))
                else:
                    # Fallback: MD5 of raw pixels (misses compressed duplicates)
                    img_resized = img.convert("L").resize((32, 32))
                    h = hashlib.md5(np.array(img_resized).tobytes()).hexdigest()
            hash_map[h].append(str(img_path))
        except Exception:
            pass

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    print(f"\n[Duplicates] Found {len(duplicates)} groups of duplicate images:")
    total_dup = 0
    for h, paths in list(duplicates.items())[:10]:  # show first 10 groups
        print(f"  Hash {h[:16]}…: {len(paths)} files")
        for p in paths:
            print(f"    {p}")
        total_dup += len(paths) - 1
    print(f"  → {total_dup} redundant files could be removed")
    return duplicates


# ─────────────────────────────────────────────────────────────────────────────
# 3. Visualize class distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_class_distribution(data_dir: str, output_dir: str = "audit_report"):
    """
    Plot class distribution for train, val, test folders side-by-side.
    Also prints imbalance ratio and recommends class weights.
    """
    os.makedirs(output_dir, exist_ok=True)
    splits = ["train", "val", "test"]
    all_counts = {}

    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path):
            continue
        counts = {}
        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                counts[cls] = len(list(Path(cls_path).glob("*.*")))
        all_counts[split] = counts

    if not all_counts:
        # data_dir IS the split (e.g. ../data/train directly)
        counts = {}
        for cls in os.listdir(data_dir):
            cls_path = os.path.join(data_dir, cls)
            if os.path.isdir(cls_path):
                counts[cls] = len(list(Path(cls_path).glob("*.*")))
        all_counts["dataset"] = counts

    fig, axes = plt.subplots(1, len(all_counts), figsize=(6 * len(all_counts), 5))
    if len(all_counts) == 1:
        axes = [axes]

    # Imbalance analysis
    print("\n[Class Distribution]")
    for ax, (split, counts) in zip(axes, all_counts.items()):
        classes = list(counts.keys())
        values  = list(counts.values())
        total   = sum(values)
        max_c   = max(values)
        min_c   = min(values)
        imbalance_ratio = max_c / (min_c + 1e-9)

        print(f"\n  {split.upper()} (total={total}, imbalance ratio={imbalance_ratio:.1f}×):")
        for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100
            rec_weight = total / (len(classes) * cnt)
            print(f"    {cls:12s}: {cnt:4d} ({pct:5.1f}%)  recommended_weight={rec_weight:.3f}")

        colors = ["#2ecc71" if v == max_c else
                  "#e74c3c" if v == min_c else "#3498db" for v in values]
        bars = ax.bar(classes, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{split} (n={total})\nImbalance {imbalance_ratio:.1f}×", fontsize=12)
        ax.set_ylabel("Count")
        ax.set_xlabel("Class")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(val), ha="center", va="bottom", fontsize=9)

    plt.suptitle("Cervical Cancer Dataset — Class Distribution", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Saved: {out_path}")
    return all_counts


# ─────────────────────────────────────────────────────────────────────────────
# 4. Find potentially mislabeled samples
# ─────────────────────────────────────────────────────────────────────────────
def find_mislabeled_samples(data_dir: str, model_checkpoint: str,
                             device_str: str = "cpu",
                             top_k: int = 20,
                             output_dir: str = "audit_report") -> list:
    """
    Strategy: run trained model on training set and flag samples where
    model predicted a DIFFERENT class with very HIGH confidence.
    These are often mislabeled or ambiguous.

    Requires a trained model checkpoint.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.hybrid_model import HerlevHybridClassifier

    device = torch.device(device_str)
    model = HerlevHybridClassifier(num_classes=5, pretrained=False)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False,
                                           num_workers=2)

    all_paths     = [s[0] for s in dataset.samples]
    all_labels    = [s[1] for s in dataset.samples]
    all_preds     = []
    all_confs     = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            probs  = F.softmax(model(images), dim=1)
            conf, pred = probs.max(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_confs.extend(conf.cpu().numpy())

    # Flag: predicted ≠ true AND confidence > threshold
    suspects = []
    for path, true_lbl, pred_lbl, conf in zip(all_paths, all_labels, all_preds, all_confs):
        if pred_lbl != true_lbl and conf > 0.80:
            suspects.append({
                "path": path,
                "true_class": CLASS_NAMES[true_lbl],
                "predicted_class": CLASS_NAMES[pred_lbl],
                "confidence": float(conf),
            })

    suspects.sort(key=lambda x: -x["confidence"])
    print(f"\n[Potentially Mislabeled] Top {min(top_k, len(suspects))} suspects:")
    for s in suspects[:top_k]:
        print(f"  conf={s['confidence']:.3f}  true={s['true_class']:8s}  "
              f"pred={s['predicted_class']:8s}  {s['path']}")

    # Visualise top suspects
    os.makedirs(output_dir, exist_ok=True)
    n_show = min(top_k, len(suspects))
    if n_show > 0:
        cols = 5
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten() if rows > 1 else [axes] * cols

        for ax, s in zip(axes, suspects[:n_show]):
            img = Image.open(s["path"]).convert("RGB").resize((128, 128))
            ax.imshow(img)
            ax.set_title(
                f"True: {s['true_class']}\nPred: {s['predicted_class']}\n"
                f"Conf: {s['confidence']:.2f}",
                fontsize=7, color="red"
            )
            ax.axis("off")

        for ax in axes[n_show:]:
            ax.axis("off")

        plt.suptitle("Potentially Mislabeled Samples (high-confidence disagreement)", fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(output_dir, "mislabeled_suspects.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → Saved: {out_path}")

    return suspects


# ─────────────────────────────────────────────────────────────────────────────
# 5. Image resolution stats
# ─────────────────────────────────────────────────────────────────────────────
def analyze_resolutions(data_dir: str):
    data_path = Path(data_dir)
    widths, heights = [], []
    for img_path in data_path.rglob("*.png"):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            pass
    for img_path in data_path.rglob("*.jpg"):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            pass

    if widths:
        print(f"\n[Resolution Stats]")
        print(f"  Width  — min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
        print(f"  Height — min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")
        unique_sizes = len(set(zip(widths, heights)))
        print(f"  Unique (W×H) combinations: {unique_sizes}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   type=str, required=True)
    parser.add_argument("--output",     type=str, default="audit_report")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint for mislabel detection")
    parser.add_argument("--device",     type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("DATASET AUDIT")
    print("=" * 60)

    find_corrupted_images(args.data_dir)
    find_duplicates(args.data_dir)
    plot_class_distribution(args.data_dir, args.output)
    analyze_resolutions(args.data_dir)

    if args.checkpoint:
        find_mislabeled_samples(
            args.data_dir, args.checkpoint,
            device_str=args.device,
            output_dir=args.output,
        )
