"""Debug script to diagnose training issues - verify data loading and model behavior."""

import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from collections import Counter

from config import CLASS_NAMES, NUM_CLASSES
from dataset import build_train_transform, build_eval_transform
from preprocessing import discover_samples, filter_quality_and_duplicates, split_samples, class_distribution


def main():
    parser = argparse.ArgumentParser(description="Debug data loading and class distribution")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to Herlev dataset")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    print("\n" + "="*70)
    print("CLASS CONFIGURATION")
    print("="*70)
    print(f"CLASS_NAMES     : {CLASS_NAMES}")
    print(f"NUM_CLASSES     : {NUM_CLASSES}")
    print()

    # Discover samples
    print("="*70)
    print("DATA DISCOVERY")
    print("="*70)
    try:
        class_names, samples = discover_samples(args.data_dir, class_names=CLASS_NAMES)
        print(f"Discovered classes: {list(class_names)}")
        print(f"Total samples found: {len(samples)}")
        print()
    except Exception as e:
        print(f"ERROR discovering samples: {e}")
        return

    # Filter quality
    print("="*70)
    print("DATA FILTERING")
    print("="*70)
    try:
        samples, duplicates, quality_issues = filter_quality_and_duplicates(samples)
        print(f"After filtering:")
        print(f"  Valid samples     : {len(samples)}")
        print(f"  Duplicates found  : {len(duplicates)}")
        print(f"  Quality issues    : {len(quality_issues)}")
        print()
    except Exception as e:
        print(f"ERROR filtering: {e}")
        return

    # Split data
    print("="*70)
    print("DATA SPLITTING")
    print("="*70)
    try:
        train_samples, val_samples, _ = split_samples(samples, val_size=0.2, seed=42)
        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")
        print()
    except Exception as e:
        print(f"ERROR splitting: {e}")
        return

    # Show class distribution
    print("="*70)
    print("TRAIN SET CLASS DISTRIBUTION")
    print("="*70)
    train_dist = class_distribution(train_samples, class_names=class_names)
    for cls_name, count in train_dist.items():
        pct = 100.0 * count / len(train_samples)
        print(f"  {cls_name:12} : {count:4d} ({pct:5.1f}%)")
    print(f"  TOTAL         : {len(train_samples):4d}")
    print()

    print("="*70)
    print("VAL SET CLASS DISTRIBUTION")
    print("="*70)
    val_dist = class_distribution(val_samples, class_names=class_names)
    for cls_name, count in val_dist.items():
        pct = 100.0 * count / len(val_samples)
        print(f"  {cls_name:12} : {count:4d} ({pct:5.1f}%)")
    print(f"  TOTAL         : {len(val_samples):4d}")
    print()

    # Verify label mapping
    print("="*70)
    print("LABEL MAPPING")
    print("="*70)
    for i, cls_name in enumerate(CLASS_NAMES):
        print(f"  Index {i}: {cls_name}")
    print()

    # Test data loading
    print("="*70)
    print("DATA LOADING TEST")
    print("="*70)
    from dataset import HerlevDataset
    from torch.utils.data import DataLoader

    dataset = HerlevDataset(train_samples[:min(32, len(train_samples))], transform=build_train_transform(args.image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    try:
        batch_idx = 0
        label_counts = Counter()
        for batch in loader:
            images = batch["image"]
            labels = batch["label"]
            
            print(f"Batch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Images dtype: {images.dtype}")
            print(f"  Images min/max: {images.min().item():.3f} / {images.max().item():.3f}")
            print(f"  Labels: {labels.tolist()}")
            print(f"  Label names: {[CLASS_NAMES[int(l)] for l in labels]}")
            
            for label in labels:
                label_counts[int(label)] += 1
            batch_idx += 1
            
        print()
        print("Label distribution in loaded batches:")
        for i, count in sorted(label_counts.items()):
            print(f"  Class {i} ({CLASS_NAMES[i]:12}): {count} samples")
        print()
        
    except Exception as e:
        print(f"ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    print("✓ All checks passed!")
    print()


if __name__ == "__main__":
    main()
