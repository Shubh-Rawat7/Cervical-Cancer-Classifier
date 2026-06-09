"""Shared configuration for the Herlev cervical cell classification stack."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

CLASS_NAMES = ["Normal", "CIN1", "CIN2", "CIN3", "Cancer"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE_CHOICES = (224,)
DEFAULT_IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 224))
if DEFAULT_IMAGE_SIZE not in IMAGE_SIZE_CHOICES:
	DEFAULT_IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATA_DIR = Path(os.environ.get("DATA_DIR", str(PROJECT_ROOT / "data")))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", str(BASE_DIR / "Checkpoints")))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(CHECKPOINT_DIR / "best_model.pt")))

DEFAULT_BACKBONES = [
	"vim_base_patch16_224",
	"mambavision_base",
	"mambaout_base_plus_rw.sw_in12k_ft_in1k",
	"mambaout_small_rw.sw_in1k",
	"convnextv2_base",
]

BACKBONE_ALIASES = {
	"vim_base_patch16_224": ["vim_base_patch16_224", "vim_tiny_patch16_224"],
	"mambavision_base": ["mambavision_base", "mambavision_small"],
	"mambaout_base_plus_rw.sw_in12k_ft_in1k": ["mambaout_base_plus_rw.sw_in12k_ft_in1k", "mambaout_small_rw.sw_in12k_ft_in1k"],
	"convnextv2_base": ["convnextv2_base", "convnextv2_tiny"],
}


@dataclass(frozen=True)
class TrainDefaults:
	epochs: int = 90
	batch_size: int = 16
	lr: float = 5e-5
	weight_decay: float = 1e-4
	warmup_epochs: int = 5
	patience: int = 15
	grad_clip: float = 1.0
	accumulation_steps: int = 2
	num_workers: int = 4
	val_split: float = 0.2
	test_split: float = 0.0
	seed: int = 42
	loss_type: str = "class_balanced_focal"
	gamma: float = 2.0
	beta: float = 0.9999
	label_smoothing: float = 0.05
	image_size: int = DEFAULT_IMAGE_SIZE
	backbone: str = "vim_base_patch16_224"
	activation: str = "silu"
	use_amp: bool = True


DEFAULTS = TrainDefaults()