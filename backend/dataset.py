"""Herlev Cervical Cell dataset, preprocessing, and augmentation utilities."""

from __future__ import annotations

import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import CLASS_NAMES, DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from preprocessing import (
    ImageRecord,
    class_distribution,
    discover_samples,
    filter_quality_and_duplicates,
    split_samples,
    undersample_samples,
)


class HerlevDataset(Dataset):
    def __init__(self, samples: Sequence[ImageRecord], transform: A.Compose | None = None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb)

        if self.transform is not None:
            image_tensor = self.transform(image=array)["image"]
        else:
            image_tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(CLASS_NAMES.index(sample.label), dtype=torch.long)
        return {
            "image": image_tensor,
            "label": label,
            "path": str(sample.path),
        }


HybridDataset = HerlevDataset
CervicalCancerDataset = HerlevDataset


def build_preprocess_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_train_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.MedianBlur(blur_limit=3, p=0.2),
            A.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05),
                interpolation=cv2.INTER_CUBIC,
                p=0.35,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.35),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.08), rotate=(-12, 12), shear=(-8, 8), p=0.6),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_eval_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    return build_preprocess_transform(image_size=image_size)


def build_tta_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> List[A.Compose]:
    base = build_eval_transform(image_size=image_size)
    common = [
        A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    tta = [
        base,
        A.Compose(common + [A.HorizontalFlip(p=1.0)]),
        A.Compose(common + [A.VerticalFlip(p=1.0)]),
        A.Compose(common + [A.Rotate(limit=20, p=1.0)]),
        A.Compose(common + [A.Affine(scale=(0.95, 1.05), translate_percent=(0.03, 0.03), rotate=(-10, 10), shear=(-6, 6), p=1.0)]),
        A.Compose(common + [A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0)]),
        A.Compose(common + [A.GaussNoise(var_limit=(5.0, 20.0), p=1.0)]),
        A.Compose(common + [A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=12, p=1.0)]),
    ]
    return tta


def build_dataloaders(
    data_dir: str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.2,
    test_split: float = 0.0,
    seed: int = 42,
    under_sample: bool = True,
    under_sample_strategy: str = "random",
    train_transform: A.Compose | None = None,
    eval_transform: A.Compose | None = None,
) -> Tuple[DataLoader, DataLoader, List[str], Dict[str, int], List[ImageRecord], List[ImageRecord]]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    class_names, samples = discover_samples(root, class_names=CLASS_NAMES)
    samples = [sample for sample in samples if sample.label in class_names]
    samples, duplicates, quality_issues = filter_quality_and_duplicates(samples)
    if quality_issues:
        print(f"Removed {len(quality_issues)} corrupted or low-quality images")
    if duplicates:
        print(f"Found {len(duplicates)} duplicate groups; keeping first occurrence only")

    train_samples, val_samples, _ = split_samples(samples, val_size=val_split, test_size=test_split, seed=seed)
    if under_sample:
        train_samples = undersample_samples(train_samples, strategy=under_sample_strategy, seed=seed)

    train_transform = train_transform or build_train_transform(image_size=image_size)
    eval_transform = eval_transform or build_eval_transform(image_size=image_size)

    train_dataset = HerlevDataset(train_samples, transform=train_transform)
    val_dataset = HerlevDataset(val_samples, transform=eval_transform)

    workers = max(0, min(int(num_workers), os.cpu_count() or 0))
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )

    train_counts = Counter(sample.label for sample in train_samples)
    train_counts = {class_name: train_counts.get(class_name, 0) for class_name in class_names}
    return train_loader, val_loader, class_names, train_counts, train_samples, val_samples


def summarize_samples(samples: Sequence[ImageRecord]) -> Dict[str, int]:
    return class_distribution(samples, class_names=CLASS_NAMES)
