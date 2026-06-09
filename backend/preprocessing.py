"""Herlev dataset preprocessing, analysis, and sampling utilities."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
from sklearn.model_selection import train_test_split

try:
    import imagehash
except Exception:  # pragma: no cover - optional dependency
    imagehash = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency for headless envs
    plt = None
    sns = None

from config import CLASS_NAMES, DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str


def list_image_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(sorted(root.rglob(f"*{ext}")))
        files.extend(sorted(root.rglob(f"*{ext.upper()}")))
    return sorted({path for path in files if path.is_file()})


def discover_class_directories(root: Path, class_names: Sequence[str] = CLASS_NAMES) -> List[str]:
    if (root / "train").exists() or (root / "val").exists() or (root / "test").exists():
        candidates = set()
        for split_name in ("train", "val", "test"):
            split_root = root / split_name
            if split_root.exists():
                candidates.update({child.name for child in split_root.iterdir() if child.is_dir()})
        return [name for name in class_names if name in candidates]

    candidates = {child.name for child in root.iterdir() if child.is_dir()}
    ordered = [name for name in class_names if name in candidates]
    if ordered:
        return ordered
    return sorted(candidates)


def discover_samples(root: str | Path, class_names: Sequence[str] = CLASS_NAMES) -> Tuple[List[str], List[ImageRecord]]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Data directory not found: {root_path}")

    discovered_classes = discover_class_directories(root_path, class_names=class_names)
    if not discovered_classes:
        raise FileNotFoundError(f"No class folders found under {root_path}")

    samples: List[ImageRecord] = []
    split_root_names = ["train", "val", "test"]

    if any((root_path / split_name).exists() for split_name in split_root_names):
        for split_name in split_root_names:
            split_root = root_path / split_name
            if not split_root.exists():
                continue
            for class_name in discovered_classes:
                class_dir = split_root / class_name
                if not class_dir.exists():
                    continue
                for path in list_image_files(class_dir):
                    samples.append(ImageRecord(path=path, label=class_name))
    else:
        for class_name in discovered_classes:
            class_dir = root_path / class_name
            if not class_dir.exists():
                continue
            for path in list_image_files(class_dir):
                samples.append(ImageRecord(path=path, label=class_name))

    if not samples:
        raise FileNotFoundError(f"No images found under {root_path}")

    return discovered_classes, samples


def split_samples(
    samples: Sequence[ImageRecord],
    val_size: float = 0.2,
    test_size: float = 0.0,
    seed: int = 42,
) -> Tuple[List[ImageRecord], List[ImageRecord], List[ImageRecord]]:
    if not samples:
        return [], [], []

    labels = [sample.label for sample in samples]
    train_samples: List[ImageRecord]
    temp_samples: List[ImageRecord]

    if test_size > 0:
        train_samples, temp_samples = train_test_split(
            list(samples),
            test_size=val_size + test_size,
            random_state=seed,
            stratify=labels,
        )
        temp_labels = [sample.label for sample in temp_samples]
        relative_val = val_size / (val_size + test_size)
        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=1.0 - relative_val,
            random_state=seed,
            stratify=temp_labels,
        )
        return list(train_samples), list(val_samples), list(test_samples)

    train_samples, val_samples = train_test_split(
        list(samples),
        test_size=val_size,
        random_state=seed,
        stratify=labels,
    )
    return list(train_samples), list(val_samples), []


def _image_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _perceptual_hash(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if imagehash is not None:
            return str(imagehash.phash(image))
        return _image_digest(path)


def verify_image_quality(path: str | Path, min_size: int = 128) -> Tuple[bool, str]:
    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.width < min_size or image.height < min_size:
                return False, f"resolution too small: {image.width}x{image.height}"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, f"corrupted or unreadable: {exc}"
    return True, "ok"


def detect_duplicates(samples: Sequence[ImageRecord]) -> Dict[str, List[Path]]:
    hashes: Dict[str, List[Path]] = {}
    for sample in samples:
        try:
            digest = _perceptual_hash(sample.path)
        except Exception:
            digest = _image_digest(sample.path)
        hashes.setdefault(digest, []).append(sample.path)
    return {digest: paths for digest, paths in hashes.items() if len(paths) > 1}


def filter_quality_and_duplicates(
    samples: Sequence[ImageRecord],
    min_size: int = 128,
) -> Tuple[List[ImageRecord], Dict[str, List[Path]], List[Tuple[Path, str]]]:
    kept: List[ImageRecord] = []
    issues: List[Tuple[Path, str]] = []
    for sample in samples:
        ok, reason = verify_image_quality(sample.path, min_size=min_size)
        if ok:
            kept.append(sample)
        else:
            issues.append((sample.path, reason))

    duplicates = detect_duplicates(kept)
    duplicate_paths = {path for paths in duplicates.values() for path in paths[1:]}
    unique = [sample for sample in kept if sample.path not in duplicate_paths]
    return unique, duplicates, issues


def preprocess_image_array(image: np.ndarray, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.dtype != np.float32 else image

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    image = cv2.medianBlur(image, 3)
    return image


def load_and_preprocess_image(path: str | Path, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb)
    return preprocess_image_array(array, image_size=image_size)


def compute_image_statistics(samples: Sequence[ImageRecord], image_size: int = DEFAULT_IMAGE_SIZE) -> Dict[str, float]:
    if not samples:
        return {}

    means = []
    stds = []
    sharpness = []
    entropy = []

    for sample in samples:
        image = load_and_preprocess_image(sample.path, image_size=image_size).astype(np.float32) / 255.0
        means.append(image.mean(axis=(0, 1)))
        stds.append(image.std(axis=(0, 1)))
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        entropy.append(float(-(hist * np.log2(hist + 1e-8)).sum()))

    channel_means = np.stack(means)
    channel_stds = np.stack(stds)
    stats = {
        "mean_r": float(channel_means[:, 0].mean()),
        "mean_g": float(channel_means[:, 1].mean()),
        "mean_b": float(channel_means[:, 2].mean()),
        "std_r": float(channel_stds[:, 0].mean()),
        "std_g": float(channel_stds[:, 1].mean()),
        "std_b": float(channel_stds[:, 2].mean()),
        "sharpness_mean": float(np.mean(sharpness)),
        "entropy_mean": float(np.mean(entropy)),
    }
    return stats


def class_distribution(samples: Sequence[ImageRecord], class_names: Sequence[str] = CLASS_NAMES) -> Dict[str, int]:
    distribution = {class_name: 0 for class_name in class_names}
    for sample in samples:
        distribution[sample.label] = distribution.get(sample.label, 0) + 1
    return distribution


def _descriptor_vector(path: Path) -> np.ndarray:
    image = load_and_preprocess_image(path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    features = np.array(
        [
            image.mean() / 255.0,
            image.std() / 255.0,
            float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 1000.0,
            float(cv2.Canny(gray, 80, 160).mean()) / 255.0,
            float(hsv[:, :, 1].mean()) / 255.0,
            float(hsv[:, :, 2].mean()) / 255.0,
            float(-(hist * np.log2(hist + 1e-8)).sum()) / 10.0,
        ],
        dtype=np.float32,
    )
    return features


def undersample_samples(
    samples: Sequence[ImageRecord],
    strategy: str = "random",
    target_count: int | None = None,
    seed: int = 42,
) -> List[ImageRecord]:
    by_class: Dict[str, List[ImageRecord]] = {}
    for sample in samples:
        by_class.setdefault(sample.label, []).append(sample)

    if not by_class:
        return []

    if target_count is None:
        target_count = min(len(items) for items in by_class.values())

    rng = random.Random(seed)
    selected: List[ImageRecord] = []
    minority_vectors = None
    minority_centroid = None

    if strategy.lower() == "nearmiss":
        minority_samples = [sample for label, items in by_class.items() if len(items) == target_count for sample in items]
        if minority_samples:
            minority_vectors = np.stack([_descriptor_vector(sample.path) for sample in minority_samples])
            minority_centroid = minority_vectors.mean(axis=0)

    for label, items in by_class.items():
        if len(items) <= target_count:
            selected.extend(items)
            continue

        if strategy.lower() == "nearmiss" and minority_centroid is not None:
            vectors = np.stack([_descriptor_vector(sample.path) for sample in items])
            distances = np.linalg.norm(vectors - minority_centroid[None, :], axis=1)
            order = np.argsort(distances)
            chosen = [items[idx] for idx in order[:target_count]]
        else:
            buckets: Dict[int, List[ImageRecord]] = {}
            for sample in items:
                bucket = int(_descriptor_vector(sample.path).sum() * 1000) % max(target_count, 4)
                buckets.setdefault(bucket, []).append(sample)
            chosen = []
            ordered_buckets = sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0]))
            while len(chosen) < target_count and ordered_buckets:
                progressed = False
                for _, bucket_items in ordered_buckets:
                    if not bucket_items or len(chosen) >= target_count:
                        continue
                    chosen.append(bucket_items.pop(rng.randrange(len(bucket_items))))
                    progressed = True
                ordered_buckets = [(key, bucket_items) for key, bucket_items in ordered_buckets if bucket_items]
                if not progressed:
                    break
            if len(chosen) < target_count:
                remaining = [sample for sample in items if sample not in chosen]
                rng.shuffle(remaining)
                chosen.extend(remaining[: target_count - len(chosen)])

        selected.extend(chosen[:target_count])

    rng.shuffle(selected)
    return selected


def sample_grid_paths(samples: Sequence[ImageRecord], per_class: int = 3) -> List[ImageRecord]:
    chosen: List[ImageRecord] = []
    for class_name in CLASS_NAMES:
        class_items = [sample for sample in samples if sample.label == class_name]
        chosen.extend(class_items[:per_class])
    return chosen


def plot_class_distribution(
    distribution: Dict[str, int],
    output_path: str | Path,
    title: str = "Herlev Class Distribution",
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = list(distribution.keys())
    values = [distribution[label] for label in labels]
    plt.figure(figsize=(8, 5))
    if sns is not None:
        sns.barplot(x=labels, y=values, palette="viridis")
    else:
        plt.bar(labels, values, color="#4C78A8")
    plt.title(title)
    plt.ylabel("Images")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close()


def plot_sample_grid(samples: Sequence[ImageRecord], output_path: str | Path, image_size: int = DEFAULT_IMAGE_SIZE) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    per_class = {}
    for sample in samples:
        per_class.setdefault(sample.label, []).append(sample)

    rows = len(CLASS_NAMES)
    cols = max(1, max((len(per_class.get(class_name, [])) for class_name in CLASS_NAMES), default=1))
    cols = min(cols, 3)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row, class_name in enumerate(CLASS_NAMES):
        class_items = per_class.get(class_name, [])[:cols]
        for col in range(cols):
            axis = axes[row, col]
            axis.axis("off")
            if col < len(class_items):
                image = load_and_preprocess_image(class_items[col].path, image_size=image_size)
                axis.imshow(image)
                axis.set_title(class_name if col == 0 else "")
    plt.tight_layout()
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close()


def normalize_images_for_training(image: np.ndarray) -> np.ndarray:
    normalized = image.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return (normalized - mean) / std
