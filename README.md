# Cervical Cancer Stage Classification

This repository implements a Herlev cervical cell classification pipeline using real microscopy images and a PyTorch backend. The current training flow is optimized for Kaggle and saves reusable checkpoints and metrics after each run.

## Classes

- Normal
- CIN1
- CIN2
- CIN3
- Cancer

## Pipeline

1. Preprocess the Herlev dataset with quality checks, duplicate detection, RGB conversion, 224×224 resizing, CLAHE, and median filtering.
2. Analyze the cleaned dataset with class distribution and sample visualizations.
3. Prepare class-balanced training and validation splits in `train/` and `val/` folders.
4. Train the hybrid image + handcrafted-feature model with augmentation, mixed precision, class-balanced focal loss, SWA, and a staged head/backbone training schedule.
5. Run test-time augmentation (TTA) inference and use the FastAPI service for production predictions.

## Main Files

- [backend/preprocessing.py](backend/preprocessing.py) - dataset cleaning, duplicate detection, and preprocessing helpers
- [backend/dataset.py](backend/dataset.py) - Herlev dataset loader, transforms, and DataLoader builders
- [backend/models/hybrid_model.py](backend/models/hybrid_model.py) - hybrid backbone plus handcrafted-feature fusion classifier
- [backend/train.py](backend/train.py) - training script with phased head/backbone fine-tuning
- [backend/inference.py](backend/inference.py) - TTA-enabled inference utility
- [notebooks/Train_Cervical_Kaggle_v17.ipynb](notebooks/Train_Cervical_Kaggle_v17.ipynb) - Kaggle-ready training notebook that resolves the repo and dataset paths automatically
- [backend/explainability/gradcam.py](backend/explainability/gradcam.py) - Grad-CAM helpers
- [backend/explainability/shap_tools.py](backend/explainability/shap_tools.py) - handcrafted-feature SHAP helpers
- [backend/api/main.py](backend/api/main.py) - FastAPI prediction service

## Quick Start

### Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Prepare the Herlev dataset

The training data directory should contain either:

- a flat class-folder layout such as `Normal/`, `CIN1/`, `CIN2/`, `CIN3/`, `Cancer/`, or
- a split layout such as `train/Normal/`, `train/CIN1/`, ... and optionally `val/` or `test/`.

Example split layout:

```text
Herlev_dataset/
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

### Preprocess the dataset

```bash
python backend/scripts/preprocess_herlev.py --data-dir ../Herlev_dataset --output-dir ../Herlev_processed
```

### Analyze the dataset

```bash
python backend/scripts/analyze_dataset.py --data-dir ../Herlev_processed --output-dir ../artifacts/analysis
```

### Train the model

For local training, use the backend script directly:

```bash
python backend/train.py \
  --data-dir ../Herlev_processed \
  --output-dir ./Checkpoints \
  --epochs 90 \
  --batch-size 16 \
  --img-size 256 \
  --backbone tf_efficientnetv2_s \
  --dropout 0.30 \
  --lr-head 3e-4 \
  --lr-backbone 3e-5 \
  --stage1-epochs 24 \
  --stage2-epochs 26 \
  --stage3-epochs 40 \
  --cb-beta 0.9999 \
  --cb-gamma 2.0 \
  --mixup-alpha 0.10 \
  --patience 15 \
  --swa-epochs 20 \
  --swa-lr 1e-5 \
  --use-amp
```

On Kaggle, the recommended entry point is [notebooks/Train_Cervical_Kaggle_v17.ipynb](notebooks/Train_Cervical_Kaggle_v17.ipynb). It auto-detects the repo and dataset locations, patches the cloned backend for Kaggle compatibility when needed, then launches training with the same defaults.

### Run inference

```bash
python backend/inference.py ./Checkpoints/best_model.pt path/to/image.png
```

### Start the API

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Kaggle Notes

- The notebook will first look for the repository in `/kaggle/working`, `/kaggle/input`, and the current working directory. If it cannot find `backend/train.py`, it will clone the GitHub repo automatically when internet access is enabled.
- The notebook searches recursively for the directory that actually contains the five class folders and uses that as `--data-dir`.
- The notebook patches the cloned backend copy at runtime so Kaggle uses the current augmentation and training signatures even before the upstream repo is refreshed.
- The best checkpoint is selected by validation accuracy.
- A Kaggle run writes these artifacts to `/kaggle/working/Checkpoints`:
  - `best_model.pt`
  - `last_model.pt`
  - `history.json`
  - `metrics.json`
- If your Kaggle dataset is mounted at a custom path, set `DATA_DIR` in the notebook before running the training cell.

## Folder Structure

```text
backend/
  api/
  models/
  scripts/
  utils/
  config.py
  dataset.py
  inference.py
  losses.py
  preprocessing.py
  train.py
  train_kfold.py
  verify_data.py
```

## Notes

- Training is based on real Herlev images and a staged fine-tuning pipeline.
- Generated-image workflows are not part of the current training and inference pipeline.
- Checkpoints are saved in the output directory and can be loaded by `backend/inference.py` or `backend/api/main.py`.
