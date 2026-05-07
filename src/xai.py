# src/xai.py  (v2 — works with YOLO detections)
#
# Two complementary XAI outputs:
#
#  1. YOLO bounding boxes  — "where are the objects" (fast, always available)
#  2. Grad-CAM heatmap     — "what visual features drove the detection"
#                            Applied to the primary (most prominent) object's
#                            bounding box crop for a focused explanation.
#
# On CPU, Grad-CAM on a 224x224 crop runs in ~800ms–1.5s.
# The bounding box annotated image is instant (YOLO already computed it).

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
from typing import List

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ── Why we still use MobileNetV3 here ─────────────────────────────────────────
#
# YOLO is not structured for Grad-CAM — its multi-scale detection head
# makes layer attribution ambiguous and slow.
#
# Our approach: crop the primary detected object from the image,
# run MobileNetV3 on just that crop, and apply Grad-CAM to the crop.
# This gives a clean, focused heatmap of "what features within this
# object made the model confident".
#
# YOLO answers: "what is in the scene and where"
# Grad-CAM answers: "what visual features drove the key detection"
# Together they are the complete XAI output.

_WEIGHTS   = MobileNet_V3_Large_Weights.IMAGENET1K_V2
_TRANSFORM = _WEIGHTS.transforms()
_LABELS    = _WEIGHTS.meta["categories"]

torch.set_num_threads(os.cpu_count() or 4)


def _load_gradcam_model() -> torch.nn.Module:
    """Load MobileNetV3 for Grad-CAM only — kept lightweight, eval mode."""
    model = models.mobilenet_v3_large(weights=_WEIGHTS)
    model.eval()
    model.cpu()
    return model


# Lazy-load — only initialised when generate_gradcam() is first called
_gradcam_model = None

def _get_gradcam_model():
    global _gradcam_model
    if _gradcam_model is None:
        print("[XAI] Loading MobileNetV3 for Grad-CAM...")
        _gradcam_model = _load_gradcam_model()
    return _gradcam_model


# ── Crop primary object for focused Grad-CAM ──────────────────────────────────

def _crop_primary_object(
    rgb_float: np.ndarray,
    detection: List[dict],
    padding: float = 0.05,
) -> tuple[np.ndarray, tuple]:
    """
    Crop the primary detected object from the full image, with optional padding.

    Args:
        rgb_float:  [H, W, 3] float [0,1] full image
        detection:  Single detection dict from run_detection()
        padding:    Fractional padding around the bbox (default 5%)

    Returns:
        crop:       [h, w, 3] float [0,1] cropped region
        crop_coords:(x1, y1, x2, y2) actual pixel coords used for the crop
    """
    H, W = rgb_float.shape[:2]
    if detection and len(detection) > 0:
        primary_object = detection[0]

        x1, y1, x2, y2 = primary_object['bbox']
    else:
        x1, y1, x2, y2 = 0, 0, W, H

    # Add padding
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(W, int(x2 + pad_x))
    y2 = min(H, int(y2 + pad_y))

    crop = rgb_float[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


# ── Grad-CAM on crop ───────────────────────────────────────────────────────────

def generate_gradcam(
    rgb_float: np.ndarray,
    primary_detection: List[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a Grad-CAM heatmap focused on the primary detected object.

    Args:
        rgb_float:          [H, W, 3] float full image
        primary_detection:  The top detection dict from run_detection()

    Returns:
        heatmap_full:  [H, W] float heatmap mapped back to full image size
        crop_overlay:  [h, w, 3] uint8 — heatmap overlay on the cropped object
        full_overlay:  [H, W, 3] uint8 — heatmap overlay on the full image
    """
    t0 = time.time()
    model = _get_gradcam_model()

    # 1. Crop the primary object
    crop_float, (cx1, cy1, cx2, cy2) = _crop_primary_object(rgb_float, primary_detection)

    # Ensure crop is large enough to process
    h, w = crop_float.shape[:2]
    if h < 32 or w < 32:
        print("[XAI] Object too small for Grad-CAM — returning empty heatmap.")
        H, W = rgb_float.shape[:2]
        empty = np.zeros((H, W), dtype=np.float32)
        fallback_overlay = np.uint8(rgb_float * 255)
        return empty, fallback_overlay, fallback_overlay

    # 2. Prepare crop tensor for MobileNetV3
    crop_pil = Image.fromarray(np.uint8(crop_float * 255))
    crop_resized = crop_pil.resize((224, 224), Image.BILINEAR)
    crop_float_224 = np.array(crop_resized).astype(np.float32) / 255.0

    input_tensor = _TRANSFORM(crop_resized).unsqueeze(0)   # [1,3,224,224]

    # 3. Run Grad-CAM on the crop
    with GradCAM(
        model=model,
        target_layers=[model.features[-1]]
    ) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=None,               # explain top class
            aug_smooth=True,
            eigen_smooth=True,
        )

    heatmap_crop = grayscale_cam[0]   # [224, 224] float [0,1]

    # 4. Overlay on crop
    crop_overlay = show_cam_on_image(
        crop_float_224,
        heatmap_crop,
        use_rgb=True,
        colormap=cv2.COLORMAP_JET,
        image_weight=0.55,
    )

    # 5. Map heatmap back to full image size
    H, W = rgb_float.shape[:2]
    heatmap_full = np.zeros((H, W), dtype=np.float32)

    heatmap_resized = cv2.resize(heatmap_crop, (cx2 - cx1, cy2 - cy1))
    heatmap_full[cy1:cy2, cx1:cx2] = heatmap_resized

    # 6. Overlay on full image
    full_overlay = show_cam_on_image(
        rgb_float,
        heatmap_full,
        use_rgb=True,
        colormap=cv2.COLORMAP_JET,
        image_weight=0.6,
    )

    elapsed = time.time() - t0
    print(f"[XAI] Grad-CAM generated in {elapsed*1000:.0f}ms")
    return heatmap_full, crop_overlay, full_overlay


# ── Focus region from heatmap ──────────────────────────────────────────────────

def get_heatmap_region(heatmap: np.ndarray) -> str:
    """
    Return a plain English description of where the heatmap peaks.
    Used by explainer.py to build the Pidgin explanation.
    """
    total = heatmap.sum()
    if total == 0:
        return "di whole image"

    H, W = heatmap.shape
    cy = float(np.sum(np.arange(H)[:, None] * heatmap) / total) / H
    cx = float(np.sum(np.arange(W)[None, :] * heatmap) / total) / W

    v = "top" if cy < 0.35 else ("bottom" if cy > 0.65 else "middle")
    h = "left" if cx < 0.35 else ("right" if cx > 0.65 else "centre")

    if v == "middle" and h == "centre":
        return "di middle part"
    if h == "centre":
        return f"di {v} part"
    if v == "middle":
        return f"di {h} side"
    return f"di {v}-{h} side"


# ── Save helpers ───────────────────────────────────────────────────────────────

def save_overlay(overlay: np.ndarray, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(overlay).save(path, quality=92)
    print(f"[XAI] Saved: {path}")
    return path


def save_comparison_plot(
    rgb_float: np.ndarray,
    heatmap_full: np.ndarray,
    full_overlay: np.ndarray,
    annotated_rgb: np.ndarray,
    detections: List[dict],
    output_path: str = "./outputs/comparison.jpg",
) -> str:
    """
    4-panel research figure:
      1. Original image
      2. YOLO detections (bounding boxes)
      3. Grad-CAM heatmap
      4. Heatmap overlay
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#1a1a1a")

    panels = [
        (rgb_float,                  "Original image"),
        (annotated_rgb / 255.0,      f"YOLO detections ({len(detections)} objects)"),
        (heatmap_full,               "Grad-CAM attention"),
        (full_overlay / 255.0,       "Attention overlay"),
    ]

    for ax, (img, title) in zip(axes, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="jet", vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.axis("off")

    # Add scene description as figure subtitle
    # fig.text(0.5, 0.01, scene["description"],
    #          ha="center", color="#cccccc", fontsize=9, wrap=True)

    plt.tight_layout(pad=1.2)
    plt.savefig(output_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[XAI] Comparison plot saved: {output_path}")
    return output_path


