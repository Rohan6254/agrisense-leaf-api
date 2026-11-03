"""
AgriSense corn leaf disease prediction API (ONNX Runtime, for Render free tier)

  GET  /health        -> status
  GET  /model-info    -> classes and test metrics (accuracy, F1 per class)
  POST /predict       -> JSON {"image_url": "https://..."}  (used by Artemis)
  POST /predict-file  -> multipart file upload (for testing / fallback)
"""
import io
import json
from pathlib import Path

import httpx
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

HERE = Path(__file__).parent
CLASSES = json.loads((HERE / "classes.json").read_text())
METRICS = json.loads((HERE / "metrics.json").read_text()) if (HERE / "metrics.json").exists() else {}
CONFIDENCE_THRESHOLD = 0.60
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

session = ort.InferenceSession(str(HERE / "corn_leaf_model.onnx"), providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name


def friendly(name: str) -> str:
    n = name.lower()
    if "rust" in n:
        return "Common Rust"
    if "gray" in n or "grey" in n or "spot" in n:
        return "Gray Leaf Spot"
    if "blight" in n:
        return "Northern Corn Leaf Blight"
    if "healthy" in n:
        return "Healthy"
    return name.replace("_", " ").title()


def preprocess(img: Image.Image) -> np.ndarray:
    # Same steps as training: resize short side to 256, center crop 224, normalize
    img = img.convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((max(256, round(w * scale)), max(256, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


app = FastAPI(title="AgriSense Corn Leaf Disease API", version="1.1.0")


class PredictRequest(BaseModel):
    image_url: str


def run_model(image_bytes: bytes) -> dict:
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="The file is not a readable image.")
    logits = session.run(None, {INPUT_NAME: preprocess(img)})[0][0]
    probs = softmax(logits)
    order = probs.argsort()[::-1]
    top = [{"label": friendly(CLASSES[i]), "confidence": round(float(probs[i]), 4)} for i in order[:3]]
    best = top[0]
    uncertain = best["confidence"] < CONFIDENCE_THRESHOLD
    return {
        "label": best["label"],
        "confidence": best["confidence"],
        "uncertain": uncertain,
        "top_3": top,
        "supported_classes": [friendly(c) for c in CLASSES],
        "note": ("Low confidence: the photo may show a problem this model was not trained on, "
                 "or the photo is unclear.") if uncertain else "",
        "model_test_macro_f1": METRICS.get("test_macro_f1"),
    }


@app.get("/health")
def health():
    return {"status": "ok", "classes": [friendly(c) for c in CLASSES]}


@app.get("/model-info")
def model_info():
    return {
        "model": METRICS.get("model", "efficientnet_b0") + " (ONNX)",
        "classes": [friendly(c) for c in CLASSES],
        "test_accuracy": METRICS.get("test_accuracy"),
        "test_macro_f1": METRICS.get("test_macro_f1"),
        "per_class": {friendly(k): v for k, v in METRICS.get("per_class", {}).items()},
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    if not req.image_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="image_url must start with http:// or https://")
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            r = await client.get(req.image_url)
            r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Could not download the image: {e}")
    if len(r.content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")
    return run_model(r.content)


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")
    return run_model(data)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")
    return run_model(data)
