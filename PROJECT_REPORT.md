# Herlev Cervical Cell Classification Project Report

## Abstract

This repository implements a Herlev cervical cell classification pipeline using a multi-scale PyTorch model. The system preprocesses microscopy images, applies augmentation, trains with focal loss and staged fine-tuning, and supports TTA inference and production deployment via FastAPI.

## 1. Introduction

The goal of this project is to classify cervical cell images into medically meaningful categories: Normal, CIN1, CIN2, CIN3, and Cancer. The repository is built around the Herlev Cervical Cell Dataset and excludes synthetic/generated-image workflows from the active training pipeline.

## 2. Dataset

### 2.1 Class labels

- Normal
- CIN1
- CIN2
- CIN3
- Cancer

### 2.2 Data layout

The active training pipeline expects a dataset root with `train/` and `val/` folders, each containing class subfolders:

```text
dataset_root/
  train/
    Normal/
    CIN1/
    CIN2/
    CIN3/
    Cancer/
  val/
    Normal/
    CIN1/
    CIN2/
    CIN3/
    Cancer/
```

### 2.3 Preprocessing pipeline

The pipeline in `backend/preprocessing.py` includes:

- image discovery and class-folder validation
- quality verification and corrupted-file detection
- duplicate detection
- RGB conversion for non-RGB inputs
- resizing to 224×224 using bicubic interpolation
- CLAHE contrast enhancement
- median blur denoising
- normalization to ImageNet mean/std

### 2.4 Split strategy

- The current training script uses explicit `train/` and `val/` folders.
- Training augmentation is applied only to the training split.
- Validation examples are used for checkpoint selection and balanced accuracy evaluation.

## 3. Model Architecture

### 3.1 Backbone

The core network is defined in `backend/models/model.py` and can use `timm` backbones such as:

- `vim_base_patch16_224`
- `mambavision_base`
- `mambaout_base_plus_rw.sw_in12k_ft_in1k`
- `mambaout_small_rw.sw_in1k`
- `convnextv2_base`

### 3.2 Feature fusion

The classifier uses backbone features with projection and fusion layers before final classification.

### 3.3 Checkpoint loading

Saved checkpoints preserve model weights and can be loaded by `backend/inference.py` or the FastAPI service.

## 4. Training Pipeline

### 4.1 Training script

`backend/train.py` implements a staged training procedure:

- Phase 1: head-only warmup
- Phase 2: partial backbone unfreeze
- Phase 3: full backbone fine-tuning

Key features:

- mixed precision training with `torch.amp`
- `AdamW` optimizer
- cosine annealing learning rate schedule
- gradient clipping
- validation-based best checkpoint saving

### 4.2 Loss functions

The training pipeline uses:

- `FocalLoss` with class-balanced weights
- class-weighted cross-entropy for evaluation

### 4.3 Augmentation

Training augmentation includes:

- random crop
- horizontal and vertical flip
- rotation
- brightness/contrast adjustment
- Gaussian noise
- affine transforms
- CLAHE and median blur

### 4.4 Activation functions and training defaults

The model supports multiple activation functions exposed as a configuration option. The default activation and core training defaults are defined in `backend/config.py` and summarized below:

- **Supported activations:** `SiLU` (default), `GELU`, `Mish`
- **Default image size:** 224 × 224
- **Default epochs:** 90
- **Default batch size:** 16
- **Optimizer:** `AdamW` with weight decay 1e-4
- **Learning rate schedule:** Cosine annealing (with warm restarts used in staged training)
- **Mixed precision:** Enabled by default (`torch.amp`)
- **Gradient clipping:** Max norm 1.0
- **Accumulation steps:** 2 (configurable)
- **Class weighting / loss:** `FocalLoss` with class-balanced weights; evaluation uses class-weighted cross-entropy where appropriate
- **Augmentation helpers:** MixUp is supported (default alpha ~0.3) and can be enabled in the training script; CutMix hooks may also be present in experimental branches

These defaults can be adjusted via the training script or by editing `backend/config.py` for reproducible experiments.

## 5. Inference and Deployment

### 5.1 Inference utility

`backend/inference.py` supports:

- single-checkpoint inference
- test-time augmentation (TTA)

### 5.2 API service

`backend/api/main.py` provides a FastAPI service with endpoints:

- `/` root metadata
- `/health` health status
- `/classes` class list and descriptions
- `/predict` image upload inference

The API loads a configured model checkpoint and returns predicted probabilities with confidence.

## 6. Scripts and Analysis Utilities

### 6.1 Dataset preprocessing script

`backend/scripts/preprocess_herlev.py`:

- discovers raw Herlev images
- cleans and deduplicates data
- performs optional quality verification
- saves preprocessed images to an output directory

### 6.2 Dataset analysis script

`backend/scripts/analyze_dataset.py` is available for dataset analysis and visualization.

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
- `backend/scripts/analyze_dataset.py`
- `README.md`
- `PROJECT_OVERVIEW.md`
- `SETUP_GUIDE.md`

## 8. Reproducibility and Notes

- Default image size is 224×224.
- Training defaults are defined in `backend/config.py`.
- The repository uses real Herlev images for training and inference.
- Generated-image workflows are excluded from active training.

## 9. Conclusions

This project provides a focused Herlev cervical cell classification pipeline with a modern training stack, TTA-enabled inference, and FastAPI deployment support.

## 10. Results & Metrics (to populate)

Populate this section with measured results from your training runs. Do not edit metric names — use the exact fields listed below so the project artifacts remain consistent.

- Overall accuracy: **--%**
- Validation balanced accuracy: **--%**
- Per-class precision / recall / F1 (Normal, CIN1, CIN2, CIN3, Cancer): **--**
- Confusion matrix: see `artifacts/confusion_matrix.png` (or a textual matrix below)

How to produce these metrics:

1. Run training and ensure `backend/train.py` writes a `history.json` or `metrics.json` into the output directory (example run):

```bash
python backend/train.py --data-dir /path/to/dataset --output-dir ./Checkpoints --epochs 90
```

2. If no metrics files are written, run the evaluation helper against the validation set (example):

```bash
python backend/scripts/evaluate_test.py --model ./Checkpoints/best_model.pt --data-dir /path/to/dataset/val --output ./artifacts/metrics.json
```

3. Use `scikit-learn` to generate a `classification_report` and `confusion_matrix` from saved predictions, then paste the numeric outputs into the fields above.

If you prefer, share the `Checkpoints/history.json` or `artifacts/metrics.json` and I will extract the numbers and update this report for you.
