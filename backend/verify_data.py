"""Utilities to verify dataset layout and basic data quality before training."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

from config import CLASS_NAMES, CHECKPOINT_DIR
from dataset import _image_files


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sample_files(files: List[Path], k: int) -> List[Path]:
    if not files:
        return []
    if k <= 0:
        return []
    step = max(1, len(files) // k)
    return [files[i] for i in range(0, len(files), step)][:k]


def verify_dataset(
    data_dir: str | Path,
    expected_classes: Optional[Iterable[str]] = None,
    min_images_per_class: int = 1,
    sample_per_class: int = 3,
    check_corrupt: bool = True,
    output_dir: Optional[str | Path] = None,
) -> Dict:
    """Verify dataset folder structure and basic image integrity.

    The function supports two layouts:
    - root/train/<class> and root/val/<class>
    - root/<class> (single-folder layout)

    It writes a JSON report to `output_dir/verify_report.json` (or CHECKPOINT_DIR if omitted)
    and saves a few thumbnail samples under `output_dir/verify_samples/`.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    out_dir = Path(output_dir) if output_dir is not None else CHECKPOINT_DIR
    _ensure_dir(out_dir)
    samples_out = out_dir / "verify_samples"
    _ensure_dir(samples_out)

    expected = list(expected_classes) if expected_classes is not None else list(CLASS_NAMES)

    report: Dict = {
        "data_root": str(root),
        "layout": None,
        "classes_found": [],
        "class_counts": {},
        "missing_classes": [],
        "corrupt_images": [],
        "sample_files": {},
    }

    train_root = root / "train"
    val_root = root / "val"

    def collect_from_dir(base: Path) -> Tuple[List[str], List[Path]]:
        classes = [d.name for d in base.iterdir() if d.is_dir()]
        files: List[Path] = []
        for cls in classes:
            files.extend(_image_files(base / cls))
        return classes, files

    if train_root.exists() and val_root.exists():
        report["layout"] = "train_val_dirs"
        class_names = [name for name in expected if (train_root / name).exists() or (val_root / name).exists()]
        report["classes_found"] = class_names

        counts = {}
        for name in class_names:
            t = _image_files(train_root / name)
            v = _image_files(val_root / name)
            counts[name] = {"train": len(t), "val": len(v), "total": len(t) + len(v)}
            sample_candidates = _sample_files(sorted(t + v), sample_per_class)
            report["sample_files"][name] = [str(p) for p in sample_candidates]
        report["class_counts"] = counts
    else:
        report["layout"] = "flat_class_dirs"
        class_names = [d.name for d in root.iterdir() if d.is_dir()]
        class_names = [name for name in expected if (root / name).exists()] or class_names
        report["classes_found"] = class_names

        counts = {}
        for name in class_names:
            files = _image_files(root / name)
            counts[name] = {"total": len(files)}
            report["sample_files"][name] = [str(p) for p in _sample_files(files, sample_per_class)]
        report["class_counts"] = counts

    missing = [c for c in expected if c not in report["classes_found"]]
    report["missing_classes"] = missing

    # Basic checks
    issues = []
    for name, cnts in report["class_counts"].items():
        total = cnts.get("total", cnts.get("total", 0))
        if total < min_images_per_class:
            issues.append(f"Class '{name}' has only {total} images (< {min_images_per_class})")

    report["warnings"] = issues

    # Corrupt image check (lightweight: try opening and verifying)
    corrupts: List[str] = []
    if check_corrupt:
        to_check = []
        for name, files in report["sample_files"].items():
            to_check.extend([Path(p) for p in files])

        # also add up to 50 random files overall if available
        all_files = []
        for name in report["classes_found"]:
            if report["layout"] == "train_val_dirs":
                t = _image_files(train_root / name)
                v = _image_files(val_root / name)
                all_files.extend(t + v)
            else:
                all_files.extend(_image_files(root / name))

        extra = [p for p in all_files if str(p) not in [str(x) for x in to_check]][:50]
        to_check.extend(extra)

        for p in to_check:
            try:
                with Image.open(p) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError, ValueError) as e:
                corrupts.append(str(p))

    report["corrupt_images"] = corrupts

    # Save sample thumbnails
    for name, paths in report["sample_files"].items():
        class_out = samples_out / name
        _ensure_dir(class_out)
        for i, p in enumerate(paths):
            try:
                img = Image.open(p).convert("RGB")
                img.thumbnail((256, 256))
                img.save(class_out / f"sample_{i}.jpg")
            except Exception:
                # ignore saving failures
                continue

    report_path = out_dir / "verify_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(Path.cwd() / "data"))
    parser.add_argument("--output-dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--min-images", type=int, default=1)
    parser.add_argument("--sample-per-class", type=int, default=3)
    parser.add_argument("--no-corrupt-check", dest="check_corrupt", action="store_false")
    args = parser.parse_args()
    rpt = verify_dataset(args.data_dir, min_images_per_class=args.min_images, sample_per_class=args.sample_per_class, check_corrupt=args.check_corrupt, output_dir=args.output_dir)
    print(json.dumps(rpt, indent=2))
