# Cervical Cancer Stage Classification

This repository implements a Herlev cervical cell classification pipeline using real microscopy images and a PyTorch backend.

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
4. Train the model with Albumentations augmentation, mixed precision, focal loss, and a staged head/backbone training schedule.
5. Run test-time augmentation (TTA) inference and use the FastAPI service for production predictions.

## Main Files

- [backend/preprocessing.py](backend/preprocessing.py) - dataset cleaning, duplicate detection, and preprocessing helpers
- [backend/dataset.py](backend/dataset.py) - Herlev dataset loader, transforms, and DataLoader builders
- [backend/models/model.py](backend/models/model.py) - multi-scale classifier backbone and fusion head
- [backend/train.py](backend/train.py) - training script with phased head/backbone fine-tuning
- [backend/inference.py](backend/inference.py) - TTA-enabled inference utility
- [backend/utils/explainability.py](backend/utils/explainability.py) - Grad-CAM and explainability helpers
- [backend/api/main.py](backend/api/main.py) - FastAPI prediction service

## Quick Start

### Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Prepare the Herlev dataset

The training data directory should contain `train/` and `val/` subfolders, each with class folders:

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

```bash
python backend/train.py --data-dir ../Herlev_processed --output-dir ./Checkpoints --epochs 90
```

### Run inference

```bash
python backend/inference.py ./Checkpoints/best_model.pt path/to/image.png
```

### Start the API

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Kaggle Notes

- When training on Kaggle, mount the dataset and point `--data-dir` to the mounted path, for example `/kaggle/input/herlevdataset`.
- If the repository is uploaded as a Kaggle Dataset, use `/kaggle/working/repo` as the working path.

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
