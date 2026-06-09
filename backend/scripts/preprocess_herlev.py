"""Preprocess and split the Herlev cervical cell dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import (
    ImageRecord,
    class_distribution,
    discover_samples,
    filter_quality_and_duplicates,
    load_and_preprocess_image,
    split_samples,
)


def _save_records(records: list[ImageRecord], output_dir: Path, image_size: int) -> None:
    for record in records:
        class_dir = output_dir / record.label
        class_dir.mkdir(parents=True, exist_ok=True)
        target = class_dir / record.path.name
        image = load_and_preprocess_image(record.path, image_size=image_size)
        import cv2

        cv2.imwrite(str(target), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the Herlev cervical cell dataset")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root Herlev dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for preprocessed output")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-only", action="store_true", help="Only clean and deduplicate; do not save split folders")
    args = parser.parse_args()

    class_names, samples = discover_samples(args.data_dir)
    cleaned_samples, duplicates, quality_issues = filter_quality_and_duplicates(samples)
    train_samples, val_samples, test_samples = split_samples(cleaned_samples, val_size=args.val_split, test_size=args.test_split, seed=args.seed)

    report = {
        "class_names": class_names,
        "input_distribution": class_distribution(samples, class_names=class_names),
        "clean_distribution": class_distribution(cleaned_samples, class_names=class_names),
        "train_distribution": class_distribution(train_samples, class_names=class_names),
        "val_distribution": class_distribution(val_samples, class_names=class_names),
        "test_distribution": class_distribution(test_samples, class_names=class_names),
        "quality_issues": len(quality_issues),
        "duplicate_groups": len(duplicates),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.clean_only:
        for split_name in ("train", "val", "test"):
            split_path = args.output_dir / split_name
            if split_path.exists():
                shutil.rmtree(split_path)
        _save_records(train_samples, args.output_dir / "train", image_size=args.image_size)
        _save_records(val_samples, args.output_dir / "val", image_size=args.image_size)
        if test_samples:
            _save_records(test_samples, args.output_dir / "test", image_size=args.image_size)

    (args.output_dir / "preprocess_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
