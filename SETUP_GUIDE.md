# Setup Guide

## 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

## 2. Prepare the Herlev Dataset

Place the raw Herlev images in `train/` and `val/` subfolders by class.

Example layout:

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

## 3. Preprocess the Dataset

```bash
python backend/scripts/preprocess_herlev.py --data-dir ../Herlev --output-dir ../Herlev_processed
```

This step performs:

- quality checks
- duplicate detection
- corrupted-file removal
- RGB conversion
- 224×224 resizing
- CLAHE
- median filtering

## 4. Analyze the Dataset

```bash
python backend/scripts/analyze_dataset.py --data-dir ../Herlev_processed --output-dir ../artifacts/analysis
```

This step generates:

- class distribution chart
- dataset statistics
- sample image grid per class

## 5. Train the Model

```bash
python backend/train.py --data-dir ../Herlev_processed --output-dir ./Checkpoints --epochs 90
```

The current training script performs staged fine-tuning over a head-only warmup, partial backbone unfreeze, and full backbone training.

## 6. Run Inference

```bash
python backend/inference.py ./Checkpoints/best_model.pt path/to/image.png
```

## 7. Start the API

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

## 8. Recommended Execution Order

1. Preprocess the raw Herlev dataset.
2. Run dataset analysis.
3. Train with the prepared `train/` and `val/` folders.
4. Review validation metrics.
5. Run inference and explainability.
6. Launch the API for deployment.

## 9. Notes

- Training uses only real Herlev images.
- Generated-image tools are not used in the current training pipeline.
- For Kaggle, mount the dataset and point `--data-dir` to the mounted path, such as `/kaggle/input/herlevdataset`.
