"""Analyze why certain classes are being filtered out during quality checks."""

import argparse
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CLASS_NAMES
from preprocessing import discover_samples, ImageRecord, list_image_files, discover_class_directories
from PIL import Image
import hashlib
import imagehash
import cv2
import numpy as np


MIN_SIZE = 128  # Default minimum size


def analyze_class_quality(root_path: Path, class_name: str):
    """Analyze quality issues for a specific class."""
    
    def get_split_roots():
        splits = {}
        for split_name in ["train", "val", "test"]:
            split_root = root_path / split_name
            if split_root.exists():
                splits[split_name] = split_root
        return splits
    
    split_roots = get_split_roots()
    if not split_roots:
        return {}
    
    results = {}
    for split_name, split_root in split_roots.items():
        class_dir = split_root / class_name
        if not class_dir.exists():
            continue
            
        print(f"\n{class_name.upper()} - {split_name.upper()} split")
        print("=" * 60)
        
        good_images = 0
        too_small = 0
        invalid_format = 0
        total = 0
        
        for path in list_image_files(class_dir):
            total += 1
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    if min(width, height) < MIN_SIZE:
                        too_small += 1
                        if too_small <= 3:  # Show first 3 examples
                            print(f"  TOO SMALL: {path.name} ({width}x{height})")
                    else:
                        good_images += 1
            except Exception as e:
                invalid_format += 1
                if invalid_format <= 3:  # Show first 3 examples
                    print(f"  INVALID: {path.name} ({type(e).__name__})")
        
        print(f"\nSummary for {class_name} ({split_name}):")
        print(f"  Total images      : {total}")
        print(f"  Good images       : {good_images} ({100*good_images/max(1,total):.1f}%)")
        print(f"  Too small         : {too_small} ({100*too_small/max(1,total):.1f}%)")
        print(f"  Invalid format    : {invalid_format} ({100*invalid_format/max(1,total):.1f}%)")
        
        results[split_name] = {
            "total": total,
            "good": good_images,
            "too_small": too_small,
            "invalid": invalid_format,
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze quality filtering issues")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to Herlev dataset")
    parser.add_argument("--min-size", type=int, default=MIN_SIZE, help="Minimum image size")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("QUALITY ISSUE ANALYSIS")
    print("=" * 70)
    print(f"Dataset location  : {args.data_dir}")
    print(f"Minimum image size: {args.min_size}x{args.min_size}")
    print()

    # Analyze each class
    class_analysis = {}
    for class_name in CLASS_NAMES:
        class_analysis[class_name] = analyze_class_quality(args.data_dir, class_name)

    # Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print()
    for class_name in CLASS_NAMES:
        print(f"\n{class_name}:")
        analysis = class_analysis.get(class_name, {})
        for split_name, stats in analysis.items():
            pct = 100 * stats["good"] / max(1, stats["total"])
            print(f"  {split_name:6} : {stats['good']:3d}/{stats['total']:3d} good ({pct:5.1f}%) | "
                  f"too_small: {stats['too_small']:3d} | invalid: {stats['invalid']:3d}")


if __name__ == "__main__":
    main()
