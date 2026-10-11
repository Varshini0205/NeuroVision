
from pathlib import Path
from contextlib import asynccontextmanager
from io import BytesIO
import sys, json, base64, traceback

import numpy as np
from PIL import Image, ImageOps
import torch
import timm
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "app" / "static"
SEG_CKPT = ROOT / "models/segmentation/hsegformer_best.pt"
CLS_CKPT = ROOT / "models/classification/convnext_tiny_4class_candidate_best.pt"
LABEL_CONFIG = ROOT / "configs/label_config.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

seg_model = None
classifier = None
startup_error = None

def png_b64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")

def extract_seg_model(loaded):
    if isinstance(loaded, torch.nn.Module):
        return loaded
    if isinstance(loaded, (tuple, list)):
        for item in loaded:
            if isinstance(item, torch.nn.Module):
                return item
    if isinstance(loaded, dict):
        for item in loaded.values():
            if isinstance(item, torch.nn.Module):
                return item
    raise TypeError("Could not extract segmentation model from loader output.")

def load_models():
    global seg_model, classifier, startup_error
    try:
        for path in (SEG_CKPT, CLS_CKPT, LABEL_CONFIG):
            if not path.is_file():
                raise FileNotFoundError(f"Required file missing: {path}")

        with LABEL_CONFIG.open("r", encoding="utf-8") as f:
            config = json.load(f)
        configured_map = {
            str(k): int(v) for k, v in config["label_to_index"].items()
        }
        if configured_map != LABEL_TO_INDEX:
            raise ValueError(f"Label config mismatch: {configured_map}")

        from src.inference.segmentation_inference import load_segmentation_model
        loaded = load_segmentation_model(ROOT, device=DEVICE)
        seg_model = extract_seg_model(loaded).to(DEVICE).eval()

        checkpoint = torch.load(
            CLS_CKPT, map_location="cpu", weights_only=False
        )
        if checkpoint.get("num_classes") != 4:
            raise ValueError("Candidate classifier must have four output classes.")

        checkpoint_map = {
            str(k): int(v)
            for k, v in checkpoint["label_to_index"].items()
        }
        if checkpoint_map != LABEL_TO_INDEX:
            raise ValueError(f"Classifier label mapping mismatch: {checkpoint_map}")

        classifier = timm.create_model(
            checkpoint["model_name"], pretrained=False, num_classes=4
        )
        classifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
        classifier.to(DEVICE).eval()
        startup_error = None
        print(f"NeuroVision V2 models loaded on {DEVICE}")
    except Exception as exc:
        seg_model = None
        classifier = None
        startup_error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

@asynccontextmanager
async def lifespan(app):
    load_models()
    yield

app = FastAPI(
    title="NeuroVision V2",
    description="Research demonstration for MRI segmentation and classification",
    version="2.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(500, "Frontend HTML file is missing.")
    return index.read_text(encoding="utf-8")

@app.get("/api/health")
def health():
    ready = seg_model is not None and classifier is not None
    return {
        "status": "ready" if ready else "error",
        "device": str(DEVICE),
        "segmentation_loaded": seg_model is not None,
        "classifier_loaded": classifier is not None,
        "classes": CLASS_NAMES,
        "error": startup_error,
    }

@app.post("/api/scan")
async def scan(file: UploadFile = File(...)):
    if seg_model is None or classifier is None:
        raise HTTPException(503, f"Models are not ready: {startup_error}")

    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if not file.filename or Path(file.filename).suffix.lower() not in allowed:
        raise HTTPException(415, "Upload a PNG, JPG, TIFF, BMP, or WEBP image.")

    content = await file.read()
    if not content:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(413, "Maximum upload size is 15 MB.")

    try:
        original = ImageOps.exif_transpose(
            Image.open(BytesIO(content))
        ).convert("L")

        if original.width < 32 or original.height < 32:
            raise HTTPException(400, "Image is too small for analysis.")
        if original.width > 4096 or original.height > 4096:
            raise HTTPException(400, "Maximum image dimensions are 4096 × 4096.")

        from src.inference.segmentation_inference import predict_segmentation
        with torch.no_grad():
            result = predict_segmentation(
                seg_model, original, DEVICE, threshold=0.5
            )

        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise RuntimeError("Unexpected segmentation inference output.")

        mask = np.asarray(result[1]).squeeze()
        if mask.ndim != 2:
            raise RuntimeError(f"Unexpected segmentation mask shape: {mask.shape}")

        mask_img = Image.fromarray(
            (mask > 0).astype(np.uint8) * 255, mode="L"
        ).resize(original.size, Image.Resampling.NEAREST)

        mask_array = np.asarray(mask_img) > 0
        mask_area = float(mask_array.mean())

        base = np.asarray(original.convert("RGB"), dtype=np.float32)
        overlay_array = base.copy()
        green = np.array([30, 238, 164], dtype=np.float32)
        overlay_array[mask_array] = (
            0.45 * base[mask_array] + 0.55 * green
        ).clip(0, 255)
        overlay = Image.fromarray(overlay_array.astype(np.uint8), mode="RGB")

        ys, xs = np.where(mask_array)
        if len(xs) and len(ys):
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            pad_x = max(4, int((x2 - x1) * 0.08))
            pad_y = max(4, int((y2 - y1) * 0.08))
            x1, x2 = max(0, x1-pad_x), min(original.width, x2+pad_x)
            y1, y2 = max(0, y1-pad_y), min(original.height, y2+pad_y)
            roi = original.crop((x1, y1, x2, y2))
            roi_method = "Predicted-mask bounding box"
        else:
            roi = original.copy()
            roi_method = "Full-image fallback (empty mask)"

        classifier_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        tensor = classifier_tf(roi).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            if DEVICE.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = classifier(tensor)
            else:
                logits = classifier(tensor)
            scores = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()

        pred_index = int(np.argmax(scores))
        pred_label = CLASS_NAMES[pred_index]

        return {
            "filename": Path(file.filename).name,
            "image_width": original.width,
            "image_height": original.height,
            "roi_width": roi.width,
            "roi_height": roi.height,
            "mask_area_fraction": mask_area,
            "roi_method": roi_method,
            "predicted_class": pred_label,
            "class_scores": [
                {"label": name, "score": float(scores[LABEL_TO_INDEX[name]])}
                for name in CLASS_NAMES
            ],
            "mri_png": png_b64(original.convert("RGB")),
            "mask_overlay_png": png_b64(overlay),
            "mask_png": png_b64(mask_img),
            "roi_png": png_b64(roi.convert("RGB")),
            "model": "ConvNeXt-Tiny · 4 classes",
            "research_notice": (
                "Research demonstration only. This result is not a medical diagnosis."
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            500, f"Analysis failed: {type(exc).__name__}: {exc}"
        ) from exc
