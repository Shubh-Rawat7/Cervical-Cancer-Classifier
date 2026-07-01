"""Herlev handcrafted feature extraction utilities.

This module extracts a 28-dimensional handcrafted feature vector composed of:

- 6 morphological features from regionprops
- 3 nuclear features derived from a provided nucleus mask
- 7 texture features from GLCM
- 12 color features from RGB and HSV statistics

The final vector is L2-normalized before being returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
from skimage.measure import label, regionprops

try:
    from skimage.feature import graycomatrix, graycoprops
except Exception as exc:  # pragma: no cover - import error is surfaced at runtime
    raise ImportError("scikit-image is required for GLCM feature extraction") from exc


FEATURE_NAMES: list[str] = [
    # Morphology (6)
    "circularity",
    "solidity",
    "eccentricity",
    "equivalent_diameter",
    "aspect_ratio",
    "extent",
    # Nucleus (3)
    "nuclear_cytoplasmic_ratio",
    "nuclear_area",
    "nuclear_perimeter",
    # Texture (7)
    "glcm_contrast",
    "glcm_correlation",
    "glcm_energy",
    "glcm_homogeneity",
    "glcm_dissimilarity",
    "glcm_entropy",
    "glcm_entropy_alt",
    # Color (12)
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "hsv_mean_h",
    "hsv_mean_s",
    "hsv_mean_v",
    "hsv_std_h",
    "hsv_std_s",
    "hsv_std_v",
]

assert len(FEATURE_NAMES) == 28, "The handcrafted feature vector must have exactly 28 dimensions"


@dataclass(frozen=True)
class FeatureVector:
    """Container for the 28-dimensional handcrafted feature vector."""

    values: np.ndarray
    names: Sequence[str] = tuple(FEATURE_NAMES)

    def as_list(self) -> list[float]:
        return self.values.astype(np.float32).tolist()


def _ensure_rgb_uint8(image: np.ndarray | "PIL.Image.Image") -> np.ndarray:
    if hasattr(image, "convert"):
        image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _resize(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)


def _safe_regionprops(mask: np.ndarray) -> object | None:
    labeled = label(mask > 0)
    props = regionprops(labeled)
    return max(props, key=lambda region: region.area) if props else None


def _normalize(value: float, scale: float, clip: bool = True) -> float:
    if not np.isfinite(value):
        return 0.0
    normalized = value / scale if scale > 0 else value
    if clip:
        normalized = float(np.clip(normalized, 0.0, 1.0))
    return float(normalized)


def _entropy_from_probs(probs: np.ndarray) -> float:
    probs = probs.astype(np.float64)
    probs = probs / (probs.sum() + 1e-12)
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values, ord=2))
    if norm <= 1e-12:
        return values.astype(np.float32)
    return (values / norm).astype(np.float32)


class CellFeatureExtractor:
    """Extract a 28-dimensional handcrafted feature vector from a cervical cell image.

    The extractor is mask-aware for nuclear measurements. If a nucleus mask is not
    provided, the nuclear features are set to zero so the API remains usable while
    keeping the implementation aligned with the Herlev mask-based methodology.
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def extract_morphology_features(self, gray: np.ndarray) -> dict[str, float]:
        """Extract 6 cell morphology features using regionprops."""
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        props = _safe_regionprops(threshold)
        if props is None:
            return {name: 0.0 for name in FEATURE_NAMES[:6]}

        minr, minc, maxr, maxc = props.bbox
        bbox_height = max(maxr - minr, 1)
        bbox_width = max(maxc - minc, 1)
        major_axis = float(getattr(props, "axis_major_length", 0.0) or 0.0)
        minor_axis = float(getattr(props, "axis_minor_length", 0.0) or 0.0)

        aspect_ratio = float(bbox_width / bbox_height)
        if major_axis > 0 and minor_axis > 0:
            aspect_ratio = float(max(major_axis, minor_axis) / max(min(major_axis, minor_axis), 1e-6))

        perimeter = float(props.perimeter or 0.0)
        area = float(props.area or 0.0)
        circularity = float((4.0 * np.pi * area) / (perimeter ** 2 + 1e-12)) if perimeter > 0 else 0.0

        return {
            "circularity": _normalize(circularity, 1.0),
            "solidity": _normalize(float(props.solidity or 0.0), 1.0),
            "eccentricity": _normalize(float(props.eccentricity or 0.0), 1.0),
            "equivalent_diameter": _normalize(float(getattr(props, "equivalent_diameter_area", 0.0) or 0.0), float(self.image_size)),
            "aspect_ratio": _normalize(aspect_ratio, 10.0),
            "extent": _normalize(float(props.extent or 0.0), 1.0),
        }

    def extract_nucleus_features(
        self,
        gray: np.ndarray,
        nucleus_mask: np.ndarray | str | Path | None = None,
    ) -> dict[str, float]:
        """Extract 3 nucleus features from a Herlev nucleus mask.

        The expected mask is a binary or grayscale nucleus mask. If no mask is
        provided, zero values are returned.
        """
        if nucleus_mask is None:
            return {
                "nuclear_cytoplasmic_ratio": 0.0,
                "nuclear_area": 0.0,
                "nuclear_perimeter": 0.0,
            }

        if isinstance(nucleus_mask, (str, Path)):
            mask = cv2.imread(str(nucleus_mask), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return {
                    "nuclear_cytoplasmic_ratio": 0.0,
                    "nuclear_area": 0.0,
                    "nuclear_perimeter": 0.0,
                }
        else:
            mask = nucleus_mask

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = np.asarray(mask)
        if mask.dtype != np.uint8:
            mask = np.clip(mask, 0, 255).astype(np.uint8)
        mask = _resize(mask, (self.image_size, self.image_size))
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        props = _safe_regionprops(mask)
        if props is None:
            return {
                "nuclear_cytoplasmic_ratio": 0.0,
                "nuclear_area": 0.0,
                "nuclear_perimeter": 0.0,
            }

        nuclear_area = float(props.area or 0.0)
        nuclear_perimeter = float(props.perimeter or 0.0)
        cytoplasm_area = max(float(gray.shape[0] * gray.shape[1]) - nuclear_area, 1.0)
        ncr = nuclear_area / cytoplasm_area

        return {
            "nuclear_cytoplasmic_ratio": _normalize(ncr, 5.0),
            "nuclear_area": _normalize(nuclear_area, float(self.image_size * self.image_size)),
            "nuclear_perimeter": _normalize(nuclear_perimeter, float(4 * self.image_size)),
        }

    def extract_texture_features(self, gray: np.ndarray) -> dict[str, float]:
        """Extract 7 texture features from GLCM statistics."""
        gray = np.asarray(gray, dtype=np.uint8)
        gray_16 = (gray // 16).astype(np.uint8)  # 16 gray levels to stabilize GLCM
        glcm = graycomatrix(
            gray_16,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=16,
            symmetric=True,
            normed=True,
        )

        contrast = float(graycoprops(glcm, "contrast").mean())
        correlation = float(graycoprops(glcm, "correlation").mean())
        energy = float(graycoprops(glcm, "energy").mean())
        homogeneity = float(graycoprops(glcm, "homogeneity").mean())
        dissimilarity = float(graycoprops(glcm, "dissimilarity").mean())

        glcm_mean = glcm.mean(axis=(0, 1, 2, 3))
        entropy = _entropy_from_probs(glcm)
        entropy_alt = _entropy_from_probs(np.sum(glcm, axis=(2, 3)).ravel())

        return {
            "glcm_contrast": _normalize(contrast, 10.0),
            "glcm_correlation": float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0)),
            "glcm_energy": _normalize(energy, 1.0),
            "glcm_homogeneity": _normalize(homogeneity, 1.0),
            "glcm_dissimilarity": _normalize(dissimilarity, 10.0),
            "glcm_entropy": _normalize(entropy, 8.0),
            "glcm_entropy_alt": _normalize(entropy_alt, 8.0),
        }

    def extract_color_features(self, image_rgb: np.ndarray) -> dict[str, float]:
        """Extract 12 color features from RGB and HSV statistics."""
        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        rgb_mean = image_rgb.reshape(-1, 3).mean(axis=0)
        rgb_std = image_rgb.reshape(-1, 3).std(axis=0)
        hsv_mean = hsv.reshape(-1, 3).mean(axis=0)
        hsv_std = hsv.reshape(-1, 3).std(axis=0)

        return {
            "rgb_mean_r": _normalize(float(rgb_mean[0]), 255.0),
            "rgb_mean_g": _normalize(float(rgb_mean[1]), 255.0),
            "rgb_mean_b": _normalize(float(rgb_mean[2]), 255.0),
            "rgb_std_r": _normalize(float(rgb_std[0]), 128.0),
            "rgb_std_g": _normalize(float(rgb_std[1]), 128.0),
            "rgb_std_b": _normalize(float(rgb_std[2]), 128.0),
            "hsv_mean_h": _normalize(float(hsv_mean[0]), 180.0),
            "hsv_mean_s": _normalize(float(hsv_mean[1]), 255.0),
            "hsv_mean_v": _normalize(float(hsv_mean[2]), 255.0),
            "hsv_std_h": _normalize(float(hsv_std[0]), 90.0),
            "hsv_std_s": _normalize(float(hsv_std[1]), 128.0),
            "hsv_std_v": _normalize(float(hsv_std[2]), 128.0),
        }

    def extract_all_features(
        self,
        image_rgb: np.ndarray,
        nucleus_mask: np.ndarray | str | Path | None = None,
    ) -> dict[str, float]:
        """Extract and return all 28 handcrafted features as a dictionary."""
        image_rgb = _resize(_ensure_rgb_uint8(image_rgb), (self.image_size, self.image_size))
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        features: dict[str, float] = {}
        features.update(self.extract_morphology_features(gray))
        features.update(self.extract_nucleus_features(gray, nucleus_mask=nucleus_mask))
        features.update(self.extract_texture_features(gray))
        features.update(self.extract_color_features(image_rgb))

        ordered = {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}
        vector = _l2_normalize(np.array([ordered[name] for name in FEATURE_NAMES], dtype=np.float32))
        return {name: float(value) for name, value in zip(FEATURE_NAMES, vector)}


def extract_medical_features(
    image: np.ndarray | "PIL.Image.Image",
    nucleus_mask: np.ndarray | str | Path | None = None,
) -> list[float]:
    """Return the 28-dimensional handcrafted feature vector for a single image."""
    extractor = CellFeatureExtractor()
    features = extractor.extract_all_features(image, nucleus_mask=nucleus_mask)
    return [features[name] for name in FEATURE_NAMES]


def extract_features_from_dataset(
    data_dir: str | Path,
    output_csv: str = "features.csv",
    mask_dir: str | Path | None = None,
) -> object:
    """Extract handcrafted features from a dataset and save them to CSV.

    If `mask_dir` is supplied, the function looks for a matching mask file using
    the same stem as the image file. This keeps the extractor aligned with the
    Herlev nucleus-mask requirement without manual segmentation.
    """
    import pandas as pd
    from tqdm import tqdm

    data_dir = Path(data_dir)
    mask_root = Path(mask_dir) if mask_dir is not None else None
    extractor = CellFeatureExtractor()
    rows: list[dict[str, float | str]] = []

    class_dirs = [child for child in data_dir.iterdir() if child.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under {data_dir}")

    for class_dir in sorted(class_dirs, key=lambda path: path.name):
        image_files = sorted(
            list(class_dir.glob("*.png"))
            + list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.bmp"))
            + list(class_dir.glob("*.tif"))
            + list(class_dir.glob("*.tiff"))
        )
        for img_path in tqdm(image_files, desc=class_dir.name):
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            nucleus_mask = None
            if mask_root is not None:
                for suffix in (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"):
                    candidate = mask_root / class_dir.name / f"{img_path.stem}{suffix}"
                    if candidate.exists():
                        nucleus_mask = candidate
                        break
                    candidate = mask_root / f"{img_path.stem}{suffix}"
                    if candidate.exists():
                        nucleus_mask = candidate
                        break

            feature_row = extractor.extract_all_features(image, nucleus_mask=nucleus_mask)
            feature_row["class"] = class_dir.name
            feature_row["filename"] = img_path.name
            rows.append(feature_row)

    df = pd.DataFrame(rows)
    output_path = data_dir.parent / output_csv
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/train")
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--output", default="train_features.csv")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        feats = extract_medical_features(dummy)
        assert len(feats) == 28, f"Expected 28 features, got {len(feats)}"
        print("Smoke test passed")
    else:
        extract_features_from_dataset(args.data_dir, args.output, mask_dir=args.mask_dir)