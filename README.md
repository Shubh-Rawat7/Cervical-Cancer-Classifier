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
4. Train the model with augmentation, mixed precision, focal loss, and a staged head/backbone training schedule.
5. Run test-time augmentation (TTA) inference and use the FastAPI service for production predictions.

## Main Files

- [backend/preprocessing.py](backend/preprocessing.py) - dataset cleaning, duplicate detection, and preprocessing helpers
- [backend/dataset.py](backend/dataset.py) - Herlev dataset loader, transforms, and DataLoader builders
- [backend/models/model.py](backend/models/model.py) - multi-scale classifier backbone and fusion head
- [backend/train.py](backend/train.py) - training script with phased head/backbone fine-tuning
- [backend/inference.py](backend/inference.py) - TTA-enabled inference utility
- [notebooks/Train_Cervical_Kaggle_v17.ipynb](notebooks/Train_Cervical_Kaggle_v17.ipynb) - Kaggle-ready training notebook that resolves the repo and dataset paths automatically
- [backend/utils/explainability.py](backend/utils/explainability.py) - Grad-CAM and explainability helpers
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
  --epochs 100 \
  --batch-size 12 \
  --img-size 256 \
  --backbone tf_efficientnetv2_m \
  --dropout 0.25 \
  --lr-head 2e-4 \
  --lr-backbone 2e-5 \
  --phase1-epochs 20 \
  --phase2-epochs 30 \
  --mixup-alpha 0.10 \
  --focal-gamma 1.2 \
  --patience 18
```

On Kaggle, the recommended entry point is [notebooks/Train_Cervical_Kaggle_v17.ipynb](notebooks/Train_Cervical_Kaggle_v17.ipynb). It auto-detects the repo and dataset locations, then launches training with the same defaults.

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
