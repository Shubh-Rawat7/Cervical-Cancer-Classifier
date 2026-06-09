# Herlev Cervical Cell Classification

This repository now implements a Herlev-only cervical cell classification pipeline that trains on real cervical cell images only.

## Classes

- Normal
- CIN1
- CIN2
- CIN3
- Cancer

## Pipeline

1. Preprocess the Herlev dataset with quality checks, duplicate removal, RGB conversion, 224×224 resizing, CLAHE, and median filtering.
2. Analyze the cleaned dataset with class distribution charts, summary statistics, and sample grids.
3. Apply intelligent under-sampling to the training split only.
4. Train with Albumentations augmentation on the training split only.
5. Fit the Herlev Mamba classifier with mixed precision, focal loss, label smoothing, cosine annealing, early stopping, and 5-fold cross-validation.
6. Run TTA, ensemble inference, confidence calibration, Grad-CAM, feature-map, and attention-map explainability.

## Main Files

- [backend/preprocessing.py](backend/preprocessing.py) - dataset cleaning, duplicate detection, under-sampling, and analysis helpers
- [backend/dataset.py](backend/dataset.py) - Herlev dataset, transforms, and DataLoader builders
- [backend/models/model.py](backend/models/model.py) - Mamba-based multi-scale classifier
- [backend/train.py](backend/train.py) - single-split and 5-fold training pipeline
- [backend/inference.py](backend/inference.py) - TTA and ensemble inference
- [backend/utils/explainability.py](backend/utils/explainability.py) - Grad-CAM, feature maps, and attention maps
- [backend/api/main.py](backend/api/main.py) - FastAPI prediction service

## Quick Start

### Install dependencies

```bash
c
pip install -r requirements.txt
```

### Preprocess the dataset

```bash
python scripts/preprocess_herlev.py --data-dir ../data/raw_herlev --output-dir ../data/processed
```

### Analyze the dataset

```bash
python scripts/analyze_dataset.py --data-dir ../data/processed --output-dir ../artifacts/analysis
```

### Train the model

```bash
python train.py --data-dir ../data/processed --output-dir ./Checkpoints --epochs 90 --use-kfold
```

### Run inference

```bash
python inference.py --model-path ./Checkpoints/best_model.pt --image-path path/to/image.png
```

### Start the API

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

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

- Training uses only real Herlev images.
- Retired generated-image workflows are no longer part of the project.
- The model is checkpointed with class names and configuration metadata for reproducible inference.
