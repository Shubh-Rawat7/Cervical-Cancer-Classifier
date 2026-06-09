# Herlev Cervical Cell Classification Project Report

## Abstract

This repository implements a Herlev-only cervical cell classification pipeline using a custom multi-scale Mamba-based neural network. The system cleans and preprocesses microscopy images, applies intelligent under-sampling and augmentation, trains on a four-class label set, and supports test-time augmentation, ensemble inference, and production deployment via FastAPI.

## 1. Introduction

The goal of this project is to classify cervical cell images into medically meaningful categories: Normal, CIN1, CIN2, and CIN3. The repository has been redesigned around the Herlev Cervical Cell Dataset, and all generated-image workflows have been retired from the active training and inference pipeline.

## 2. Dataset

### 2.1 Class labels

- Normal
- CIN1
- CIN2
- CIN3

### 2.2 Processed dataset counts

The processed dataset in `data/processed` currently contains:

- `train`: 555 images
  - CIN1: 146
  - CIN2: 115
  - CIN3: 126
  - Normal: 168
- `val`: 139 images
  - CIN1: 36
  - CIN2: 29
  - CIN3: 32
  - Normal: 42

### 2.3 Preprocessing pipeline

The pipeline in `backend/preprocessing.py` includes:

- image discovery and class-folder validation
- image quality verification and corrupted-file detection
- duplicate detection using perceptual hashing and SHA-256 fallback
- optional quality filtering with `--skip-quality-check` to preserve more valid Herlev images
- RGB conversion for grayscale, RGBA, or BGR inputs
- resizing to 224×224 using bicubic interpolation
- CLAHE contrast enhancement on LAB luminance channel
- median blur denoising
- image normalization to ImageNet mean/std

### 2.4 Split strategy

- Stratified train/validation split with default `val_split=0.2`
- Optional test split support is available but not required by default
- Under-sampling is applied only to the training split using a configurable strategy

## 3. Model Architecture

### 3.1 Backbone

The core network is defined in `backend/models/model.py` and uses the `timm` library to instantiate a feature extractor. Supported backbones include:

- `vim_base_patch16_224`
- `mambavision_base`
- `mambaout_base_plus_rw.sw_in12k_ft_in1k`
- `mambaout_small_rw.sw_in1k`
- `convnextv2_base`

Backbone resolution and available models are resolved dynamically through `timm.list_models()`.

### 3.2 Multi-scale fusion

The `HerlevMambaClassifier` performs the following steps:

- extracts multi-scale feature maps from the backbone
- projects each scale to a shared embedding dimension
- spatially pools each feature map into a token sequence
- appends a learned classification token and positional embedding
- fuses tokens through `nn.MultiheadAttention`
- refines fused tokens using Mamba mixer blocks
- generates final logits via a two-layer head with dropout and activation

### 3.3 Mamba Mixer

The project includes a fallback implementation when `mamba_ssm.Mamba` is unavailable:

- `MambaMixerBlock` wraps `LiteMambaBlock` for compatibility
- `LiteMambaBlock` uses layer norm, linear projections, depthwise conv, and gated activation
- activation functions supported: SiLU, GELU, Mish

### 3.4 Checkpoint loading

The checkpoint loader supports metadata preservation and prefix-stripped state dicts, enabling resume and inference from saved models.

## 4. Training Pipeline

### 4.1 Training script

`backend/train.py` implements both single-split and cross-validation training.

Key features:

- explicit device selection via `--device {cuda,cpu}` and legacy `--cpu` alias
- mixed precision training with `torch.amp.autocast('cuda', enabled=...)`
- `GradScaler('cuda', enabled=...)` to avoid AMP deprecation issues
- class-balanced focal loss with label smoothing
- cosine annealing learning rate schedule
- gradient accumulation and gradient clipping
- early stopping based on validation loss

### 4.2 Loss functions

`backend/losses.py` provides:

- `FocalLoss`
- `ClassBalancedFocalLoss`
- `CrossEntropyLoss` with class-balanced weighting
- effective number weighting based on sample counts

### 4.3 Augmentation

Training augmentation in `backend/dataset.py` includes:

- random crop
- horizontal and vertical flip
- rotation
- brightness/contrast adjustment
- Gaussian noise
- affine transforms

Evaluation and TTA transforms use CLAHE, median blur, and deterministic augmentations such as flips and mild rotations.

### 4.4 Cross-validation support

- `backend/train.py` can run K-fold training using `StratifiedKFold`
- `backend/train_kfold.py` is a compatibility wrapper that invokes `train.main()`

## 5. Inference and Deployment

### 5.1 Inference utility

`backend/inference.py` supports:

- single-model inference
- model checkpoint ensemble averaging
- test-time augmentation with configurable views
- temperature scaling for confidence calibration

### 5.2 API service

`backend/api/main.py` provides a FastAPI service with endpoints:

- `/` root metadata and status
- `/health` health check
- `/classes` class label list and descriptions
- `/predict` image upload inference

The API loads the first available checkpoint from configured paths and returns predicted label probabilities with confidence.

## 6. Scripts and Analysis Utilities

### 6.1 Dataset preprocessing script

`backend/scripts/preprocess_herlev.py`:

- discovers raw Herlev images
- cleans and deduplicates data
- performs optional per-image quality verification and supports `--skip-quality-check`
- splits into `train`, `val`, and optional `test`
- saves preprocessed images to the output directory
- writes `preprocess_report.json`

### 6.2 Dataset analysis script

`backend/scripts/analyze_dataset.py` is available for dataset visualization and analysis, though it is not included in this report summary.

## 7. Project Structure

- `backend/config.py`
- `backend/dataset.py`
- `backend/losses.py`
- `backend/models/model.py`
- `backend/train.py`
- `backend/inference.py`
- `backend/api/main.py`
- `backend/preprocessing.py`
- `backend/scripts/preprocess_herlev.py`
- `README.md`
- `PROJECT_OVERVIEW.md`
- `SETUP_GUIDE.md`

## 8. Reproducibility and Notes

- Default image size is 224×224
- Training defaults come from `backend/config.py`
- The repository uses only real Herlev images
- Synthetic-generation workflows remain documented in notebooks, but they are retired from active training and deployment

## 9. Conclusions

This project is a focused Herlev cervical cell classification pipeline built for reproducibility, production readiness, and explainable inference. The architecture combines a modern backbone with custom multi-scale attention and mixer blocks, while the training stack uses class-balanced losses and structured preprocessing to mitigate dataset imbalance.
