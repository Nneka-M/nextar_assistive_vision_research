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


# ── Scene description ──────────────────────────────────────────────────────────

_SCENE_SIGNATURES = {
    "market / street scene": ["person", "car", "motorcycle", "bicycle",
                               "bus", "truck", "traffic light", "handbag", "backpack"],
    "clinic / health setting": ["person", "bed", "chair", "bottle",
                                 "cup", "scissors"],
    "food scene":             ["banana", "apple", "orange", "bowl", "cup",
                                "fork", "knife", "spoon", "sandwich", "pizza",
                                "cake", "dining table"],
    "road / traffic scene":  ["car", "truck", "bus", "motorcycle", "bicycle",
                               "traffic light", "stop sign"],
    "indoor / room":         ["chair", "sofa", "bed", "tv", "laptop",
                               "book", "clock", "vase", "person"],
}

_PLURALS = {
    "person": "people", "man": "men", "woman": "women", "child": "children",
    "bus": "buses", "knife": "knives", "bench": "benches", "watch": "watches",
}

def _plural(label: str, count: int) -> str:
    return label if count == 1 else _PLURALS.get(label, label + "s")


def infer_scene_type(detections: list[dict]) -> str:
    detected = {d["label"] for d in detections}
    best_scene, best_hits = "general scene", 0
    for scene, keywords in _SCENE_SIGNATURES.items():
        hits = len(detected & set(keywords))
        if hits > best_hits:
            best_hits, best_scene = hits, scene
    return best_scene


def build_scene_description(detections: list[dict]) -> dict:
    """
    Convert raw detections into a structured scene summary.

    Returns:
        scene_type:    str  — inferred scene category
        primary:       list — top 3 most visually prominent object types
        counts:        dict — {label: count} for all detected objects
        total_objects: int
        description:   str  — clean English sentence describing the scene
    """
    if not detections:
        return {
            "scene_type":    "unknown",
            "primary":       [],
            "counts":        {},
            "total_objects": 0,
            "description":   "Nothing was detected in this image.",
        }

    counts = Counter(d["label"] for d in detections)

    # Score each label by total visual prominence: sum(area * confidence)
    prominence = {}
    for d in detections:
        lbl = d["label"]
        prominence[lbl] = prominence.get(lbl, 0) + d["area_frac"] * d["confidence"]

    primary = sorted(prominence, key=prominence.get, reverse=True)[:3]
    scene_type = infer_scene_type(detections)

    # Build object description string
    parts = [f"{counts[lbl]} {_plural(lbl, counts[lbl])}" for lbl in primary]
    if len(parts) == 1:
        object_str = parts[0]
    elif len(parts) == 2:
        object_str = f"{parts[0]} and {parts[1]}"
    else:
        object_str = f"{parts[0]}, {parts[1]}, and {parts[2]}"

    # Secondary objects
    remaining = [lbl for lbl in counts if lbl not in primary]
    if remaining:
        other_str = ", ".join(remaining[:3])
        if len(remaining) > 3:
            other_str += f" and {len(remaining) - 3} more"
        description = (
            f"This looks like a {scene_type}. "
            f"I can see {object_str}. "
            f"There are also {other_str} in the scene."
        )
    else:
        description = (
            f"This looks like a {scene_type}. "
            f"I can see {object_str}."
        )

    return {
        "scene_type":    scene_type,
        "primary":       primary,
        "counts":        dict(counts),
        "total_objects": len(detections),
        "description":   description,
    }


# ── Annotated image ────────────────────────────────────────────────────────────

def get_annotated_image(model: YOLO, image_source, conf: float = 0.30) -> np.ndarray:
    """
    Return YOLO-annotated image with bounding boxes drawn.
    Returns [H, W, 3] uint8 RGB — ready for Gradio display.
    """
    results = model(image_source, conf=conf, device="cpu", verbose=False)
    annotated_bgr = results[0].plot()
    return annotated_bgr[:, :, ::-1]   # BGR → RGB


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    model = load_model()

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("[Inference] No image given — downloading COCO test image...")
        import urllib.request
        os.makedirs("./test_images", exist_ok=True)
        image_path = "./test_images/bus.jpg"
        if not os.path.exists(image_path):
            urllib.request.urlretrieve(
                "https://ultralytics.com/images/bus.jpg", image_path
            )

    img_array, rgb_float = load_image(image_path)
    detections = run_detection(model, image_path)
    scene = build_scene_description(detections)

    print("\n── Detections ─────────────────────────────────────────")
    for d in detections:
        print(f"  {d['label']:<20} {d['confidence']*100:5.1f}%  "
              f"centre=({d['centre'][0]:.2f},{d['centre'][1]:.2f})  "
              f"area={d['area_frac']*100:.1f}%")

    print("\n── Scene Summary ──────────────────────────────────────")
    print(f"  Scene type    : {scene['scene_type']}")
    print(f"  Total objects : {scene['total_objects']}")
    print(f"  Counts        : {scene['counts']}")
    print(f"  Description   : {scene['description']}")
    print("───────────────────────────────────────────────────────")