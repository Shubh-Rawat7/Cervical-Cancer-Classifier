# Setup Guide

## 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

## 2. Prepare the Herlev Dataset

Place the raw Herlev images in a single root folder or in `train/`, `val/`, `test/` subfolders by class.

Example layout:

```text
Herlev/
  Normal/
  CIN1/
  CIN2/
  CIN3/
  Cancer/
```

## 3. Preprocess the Dataset

```bash
python scripts/preprocess_herlev.py --data-dir ../Herlev --output-dir ../Herlev_processed
```

This step performs:

- quality checks
- duplicate detection
- corrupted-file removal
- RGB conversion
- 224×224 resize
- CLAHE
- median filtering
- stratified split

## 4. Analyze the Dataset

```bash
python scripts/analyze_dataset.py --data-dir ../Herlev_processed --output-dir ../artifacts/analysis
```

This step generates:

- class distribution chart
- dataset statistics
- sample image grid per class

## 5. Train the Model

Single split:

```bash
python train.py --data-dir ../Herlev_processed --output-dir ./Checkpoints --epochs 90
```

K-fold:

```bash
python train.py --data-dir ../Herlev_processed --output-dir ./Checkpoints --use-kfold --k-folds 5
```

Activation ablation:

```bash
python train.py --data-dir ../Herlev_processed --output-dir ./Checkpoints --activation-ablation
```

## 6. Run Inference

```bash
python inference.py --model-path ./Checkpoints/best_model.pt --image-path path/to/image.png
```

## 7. Start the API

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

## 8. Recommended Execution Order

1. Preprocess the raw Herlev dataset.
2. Run dataset analysis.
3. Train with single split or K-fold.
4. Review metrics and confusion matrices.
5. Run inference and explainability.
6. Launch the API for deployment.

## 9. Notes

- Training uses only real Herlev images.
- Retired generated-image tools are no longer part of the project.
- For Colab or Kaggle, mount the dataset and point `--data-dir` to the mounted folder.
