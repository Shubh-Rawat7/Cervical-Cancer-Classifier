"""FastAPI server for Herlev cervical cell classification."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
_PROJECT = _BACKEND.parent
for path in (_HERE, _BACKEND, _PROJECT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import CLASS_NAMES, DEFAULT_IMAGE_SIZE, MODEL_PATH
from dataset import build_eval_transform
from models.model import get_class_names, load_model


app = FastAPI(title="Herlev Cervical Cell Classification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
device = None
transform = None
class_names = list(CLASS_NAMES)
model_load_error = None


@app.on_event("startup")
async def load_model_on_startup():
    global model, device, transform, class_names, model_load_error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_candidates = [
        Path(os.environ.get("MODEL_PATH", MODEL_PATH)),
        _BACKEND / "Checkpoints" / "best_model.pt",
        _BACKEND / "Checkpoints" / "last_model.pt",
        _BACKEND / "Checkpoints" / "fold_1_best.pt",
    ]
    model_path = next((candidate for candidate in checkpoint_candidates if candidate.exists()), None)

    if model_path is None:
        model_load_error = "Checkpoint not found. Train the Herlev model first."
        return

    try:
        model = load_model(model_path, device=device)
        class_names = get_class_names(model_path)
        transform = build_eval_transform(DEFAULT_IMAGE_SIZE)
        model_load_error = None
    except Exception as exc:
        model = None
        model_load_error = f"Failed to load checkpoint: {exc}"


@app.get("/")
async def root():
    return {
        "message": "Herlev Cervical Cell Classification API",
        "model_loaded": model is not None,
        "classes": class_names,
        "endpoints": {"/predict": "POST", "/health": "GET", "/classes": "GET"},
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_error": model_load_error,
        "device": str(device) if device is not None else None,
    }


@app.get("/classes")
async def get_classes_endpoint():
    return {
        "classes": class_names,
        "descriptions": {
            "Normal": "Healthy cervical tissue",
            "CIN1": "Low-grade squamous intraepithelial lesion",
            "CIN2": "Moderate dysplasia",
            "CIN3": "Severe dysplasia or carcinoma in situ",
            "Cancer": "Invasive cervical cancer",
        },
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or transform is None:
        raise HTTPException(status_code=503, detail=model_load_error or "Model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image. Got: {file.content_type}")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image=np.asarray(image))["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(probabilities[0, predicted_idx].item())

        return JSONResponse(
            content={
                "success": True,
                "predicted_class": class_names[predicted_idx],
                "confidence": round(confidence, 6),
                "probabilities": {class_names[i]: round(float(probabilities[0, i].item()), 6) for i in range(len(class_names))},
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing image: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
