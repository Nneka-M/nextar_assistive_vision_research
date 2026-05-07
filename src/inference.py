# src/inference.py  (v2 — replaces MobileNetV3 with YOLO11n)
#
# YOLO11n detects multiple objects simultaneously with bounding boxes,
# giving us a proper scene description rather than a single forced label.
#
# Install:
#   pip install ultralytics
#   (downloads yolo11n.pt automatically on first run — ~5 MB)
#
# CPU performance: ~100–200ms per image on a modern laptop CPU.

import time
import os
import numpy as np
from PIL import Image
from collections import Counter
from ultralytics import YOLO


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(model_path: str = "yolo11n.pt") -> YOLO:
    """
    Load YOLO11n. Downloads weights automatically on first run (~5 MB).
    Subsequent runs load from local cache instantly.

    YOLO11n (nano) chosen for CPU: 3–4x faster than YOLO11s
    with acceptable accuracy for scene description.
    """
    print("[Inference] Loading YOLO11n...")
    t0 = time.time()
    model = YOLO(model_path)
    elapsed = time.time() - t0
    print(f"[Inference] Model ready in {elapsed:.1f}s")
    return model


# ── Image loading ──────────────────────────────────────────────────────────────

def load_image(image_source) -> tuple[np.ndarray, np.ndarray]:
    """
    Load from file path or PIL Image.

    Returns:
        img_array:  [H, W, 3] uint8  — passed directly to YOLO
        rgb_float:  [H, W, 3] float32 in [0,1] — used for heatmap overlay
    """
    if isinstance(image_source, str):
        img = Image.open(image_source).convert("RGB")
    else:
        img = image_source.convert("RGB")

    img_array = np.array(img)
    rgb_float = img_array.astype(np.float32) / 255.0
    return img_array, rgb_float


# ── Core detection ─────────────────────────────────────────────────────────────

def run_detection(
    model: YOLO,
    image_source,
    conf_threshold: float = 0.30,
    iou_threshold: float  = 0.45,
) -> list[dict]:
    """
    Run YOLO11n and return a clean list of detected objects.

    Returns list of dicts sorted by confidence (highest first):
    [
      {
        "label":      "person",
        "confidence": 0.87,
        "bbox":       [x1, y1, x2, y2],  # pixel coords
        "centre":     (cx, cy),           # normalised 0–1
        "area_frac":  0.14,               # fraction of image area
      },
      ...
    ]
    """
    t0 = time.time()

    results = model(
        image_source,
        conf=conf_threshold,
        iou=iou_threshold,
        device="cpu",
        verbose=False,
        imgsz=640,
    )

    detections = []
    r = results[0]
    img_h, img_w = r.orig_shape

    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = model.names[cls]

        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        area_frac = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)

        detections.append({
            "label":      label,
            "confidence": conf,
            "bbox":       [x1, y1, x2, y2],
            "centre":     (cx, cy),
            "area_frac":  area_frac,
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    elapsed = time.time() - t0
    print(f"[Inference] {elapsed*1000:.0f}ms — {len(detections)} objects found")
    return detections

# ── Annotated image ────────────────────────────────────────────────────────────

def get_annotated_image(model: YOLO, image_source, conf: float = 0.30) -> np.ndarray:
    """
    Return YOLO-annotated image with bounding boxes drawn.
    Returns [H, W, 3] uint8 RGB — ready for Gradio display.
    """
    results = model(image_source, conf=conf, device="cpu", verbose=False)
    annotated_bgr = results[0].plot()
    return annotated_bgr[:, :, ::-1]   # BGR → RGB


