# Project Overview

This repository implements a Herlev cervical cell classification pipeline using a PyTorch backend. The current workflow focuses on real Herlev microscopy images and does not include synthetic or generated-image training workflows.

## Architecture

```mermaid
flowchart TD
    A[Raw Herlev images] --> B[Preprocess + cleanup]
    B --> C[Dataset analysis]
    C --> D[Train/val folder layout]
    D --> E[Albumentations augmentation]
    E --> F[Staged training (head/backbone)]
    F --> G[Best model checkpoint]
    G --> H[TTA inference]
    H --> I[FastAPI service]
```

## Directory Map

- [backend/preprocessing.py](backend/preprocessing.py) - image verification, duplicate detection, preprocessing, statistics, and plotting
- [backend/dataset.py](backend/dataset.py) - Herlev dataset loader and Albumentations transform pipelines
- [backend/models/model.py](backend/models/model.py) - multi-scale classifier backbone and fusion head
- [backend/losses.py](backend/losses.py) - focal loss and class-weighting helpers
- [backend/train.py](backend/train.py) - training script with phased fine-tuning
- [backend/inference.py](backend/inference.py) - TTA-enabled inference utility
- [backend/utils/explainability.py](backend/utils/explainability.py) - Grad-CAM and explainability helpers
- [backend/api/main.py](backend/api/main.py) - FastAPI prediction server
- [backend/scripts/preprocess_herlev.py](backend/scripts/preprocess_herlev.py) - dataset cleaning and materialization
- [backend/scripts/analyze_dataset.py](backend/scripts/analyze_dataset.py) - dataset analysis and visualizations

## Execution Order

1. Run `backend/scripts/preprocess_herlev.py` on the raw Herlev dataset.
2. Run `backend/scripts/analyze_dataset.py` on the cleaned dataset.
3. Train with `backend/train.py` using the prepared `train/` and `val/` folders.
4. Evaluate with `backend/inference.py`.
5. Use `backend/utils/explainability.py` for Grad-CAM visualization.
6. Serve predictions with `backend/api/main.py`.

## Training Strategy

- Preprocessing includes data validation, duplicate detection, RGB conversion, 224×224 resizing, CLAHE, median filtering, and ImageNet normalization.
- Training uses Albumentations augmentation on the training split only.
- The model is fine-tuned in phases: head-only warmup, partial backbone unfreeze, and full backbone fine-tuning.
- The optimizer uses `AdamW` and cosine annealing restarts.
- Loss uses focal weighting with class-balanced coefficients.

## Pipeline & Model Working

- **Data preparation:** `backend/dataset.py` loads `train/` and `val/` image folders, applies dataset quality filtering, and builds DataLoader objects.
- **Transforms:** Training uses augmentations such as random crop, flips, rotation, brightness/contrast, Gaussian noise, affine warps, CLAHE, median blur, and normalization. Validation uses deterministic preprocessing.
- **Model architecture:** `backend/models/model.py` builds a multi-scale classifier with a configurable backbone from `timm` and a fusion head. Number of output classes is controlled by `config.CLASS_NAMES`.
- **Loss and sampling:** The training pipeline uses `FocalLoss` and class weights derived from the training distribution.
- **Training loop:** `backend/train.py` runs a staged schedule with mixed precision (`torch.amp`), gradient clipping, and best-checkpoint saving based on balanced accuracy.
- **Inference:** `backend/inference.py` supports test-time augmentation (TTA) and returns averaged softmax probabilities per image.
- **Serving:** `backend/api/main.py` provides a FastAPI service for single-image prediction, health checks, and class metadata.

## Inference Strategy

- Single-checkpoint TTA inference is supported.
- Test-time augmentation averages multiple transformed views of the same image.
- API deployment is available through FastAPI.

## Removed Components

- Synthetic/generated-image training workflows
- Dataset folders that held generated samples
- Training pipelines that depended on generated images

## Notes

- The repository currently supports five class labels: Normal, CIN1, CIN2, CIN3, Cancer.
- Training expects a directory layout with `train/` and `val/` class subfolders.
