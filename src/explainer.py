# src/explainer.py  (v3 — LLM-powered, no hardcoded sentence structure)
#
# Uses a local LLM via Ollama to generate natural Pidgin English explanations.
# No hardcoded templates, no lookup tables, no rigid sentence patterns.
#
# ── Setup (one-time) ──────────────────────────────────────────────────────────
#
#   1. Install Ollama:
#      Mac/Linux:  curl -fsSL https://ollama.com/install.sh | sh
#      Windows:    Download from https://ollama.com/download
#
#   2. Pull the model (one-time download, ~800 MB):
#      ollama pull gemma3:1b
#
#   3. Ollama runs as a background service automatically after install.
#      Confirm it's running: ollama list
#
#   4. Install the Python client:
#      pip install ollama
#
# ── Tier structure ────────────────────────────────────────────────────────────
#
#   Tier 1 — Ollama local (offline, ~2–5s on CPU)   ← primary
#   Tier 2 — Claude API  (online, ~1s)               ← if ANTHROPIC_API_KEY set
#   Tier 3 — Bare-minimum fallback string            ← only if both unavailable
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import sys
from dotenv import load_dotenv
from typing import List, Dict
from collections import Counter


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(detections: List[Dict], heatmap_region: str) -> str:
    """
    Build a structured prompt from the scene dict + heatmap region.
    The LLM receives clean facts and decides how to express them.
    """
    print([type(detection) for detection in detections])
    counts   = Counter(detection.get("label", "") for detection in detections)
    # scene_type = infer_scene_type(detections)
    total    = len(detections)
    # english_desc = build_scene_description(detections).get("description", "")

    # Format object inventory
    if counts:
        obj_lines = "\n".join(
            f"  - {label}: {count}" for label, count in counts.items()
        )
    else:
        obj_lines = "  - nothing detected"

    prompt = f"""You are helping a visually impaired person understand what is in front of them.

A computer vision model analysed an image and found the following:

 
 Total objects detected: {total}
Objects detected: {obj_lines}



The AI's attention heatmap was focused on: {heatmap_region}


Your task:
Write 2–3 sentences first in natural English that:
1. Describe the scene based on the detected objects, their positions and area occupied by the objects in the image. You can also infer the scene type based on the combination of objects (e.g., "This looks like a market scene because I see many people, and foodstuffs.") using {detections}.
2. Highlight the focus of the scene (use the {heatmap_region} ) and then explain what you think is going on in that area based on the detected objects.
3. Infer the possible activities happening in the scene based on the detected objects and their relationships (e.g., "People are likely shopping and socializing in this market scene.").
4. You don't need to state the number of each object in the scene. Your goal is to infer and describe the scene in a way that is helpful and informative for a visually impaired person, not to list out object counts.
Rules:
- Write responses in English sentences, nothing else
- No preamble, no labels, no quotation marks
- Sound like a friendly, helpful person talking naturally

"""

    return prompt


# ── Tier 1: gemma (local, offline) ──────────────────────────────────────────

# def _call_gemma(prompt: str) -> str:
#     """
#     Call a locally running gemma model using transformers library.
#     """
#     from transformers import pipeline
#     import torch
#     import accelerate
#     import os
#     from dotenv import load_dotenv
#     load_dotenv()  # Load environment variables from .env file

#     t0 = time.time()
#     pipe= pipeline("text-generation", model="google/gemma-3-1b-it", device="cpu", dtype=torch.float32, token= os.getenv("HUGGINGFACE_API_KEY"))
    
#     messages =[{"role": "user", "content": prompt}]
#     response = pipe(
#         messages,
#         max_new_tokens=120,
#         temperature=1.0
#     )

#     elapsed = time.time() - t0
#     result = response["message"]["content"].strip()
#     print(f"[Explainer-Gemma] {elapsed:.1f}s — model")
#     return result


# ── Tier 2: gemini API (online fallback) ──────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    """
    Call gemini API as an online fallback.
    Requires: pip install google-genai
    """
    from google import genai
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
    client= genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    llm = client.models.generate_content(model= "gemini-2.5-flash-lite", contents= prompt)
    
    response=llm.text
    return response


def _call_groq(prompt:str) -> str:
    import os

    from groq import Groq

    client = Groq(
        # This is the default and can be omitted
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content

# ── Main entry point ──────────────────────────────────────────────────────────

def build_pidgin_from_scene(
    detections: List[Dict],
    heatmap_region: str,
    force_tier: str | None = 'gemini',
) -> str:
    
    """
    Generate a Pidgin English explanation for a detected scene.

    Args:
        detections:     List of detected objects with their properties
        heatmap_region: Output of xai.get_heatmap_region()
        force_tier:     "gemma" | "gemini" | "fallback" — bypass auto-detection
                        Leave None for automatic tier selection.

    Returns:
        Natural Pidgin English string ready for Afro-TTS.
    """
    prompt = _build_prompt(detections, heatmap_region)

    # ── Tier 1: gemma ──
    # if force_tier in (None, "gemma"):
    #     try:
    #         return _call_gemma(prompt)
    #     except Exception as e:
    #         if force_tier == "gemma":
    #             raise
    #         print(f"[Explainer] Gemma unavailable ({type(e).__name__}) — trying Gemini API...")

    # ── Tier 2: Gemini API ──
    if force_tier in (None, "gemini"):
        # api_key = "AIzaSyDJfUN04Ez9ShUxlAEj3zjCJHvWPArelnk"
        # if api_key:
            try:
                result = _call_gemini(prompt)
                print("[Explainer] Used gemini API.")
                return result
            except Exception as e:
                print(f"[Explainer] gemini API failed ({type(e).__name__}) — using fallback.")
        # else:
        #     print("[Explainer] No gemini_api_key set — skipping gemini tier.")

    # ── Tier 3: Bare fallback ──
    print("[Explainer] Using groq fallback (gemini not available).")
    return _call_groq(prompt)


# ── Backwards-compatible alias ────────────────────────────────────────────────
# xai.py calls build_explanation_template() for single-object smoke test

def build_explanation_template(
    label: str,
    confidence: float,
    heatmap_region: str,
    top3: list = None,
) -> str:
    """
    Thin wrapper — converts a single detection into a minimal scene dict
    and routes through the LLM pipeline.
    """
    scene = {
        "scene_type":    "general scene",
        "primary":       [label],
        "counts":        {label: 1},
        "total_objects": 1,
        "description":   f"A {label} detected with {int(confidence*100)}% confidence.",
    }
    return build_pidgin_from_scene(scene, heatmap_region)


# ── Ollama health check ───────────────────────────────────────────────────────

# def check_ollama(model: str = "gemma3:1b") -> dict:
    """
    Check if Ollama is running and the model is available.
    Call this at app startup to surface issues early.

    Returns:
        {"available": True/False, "model_ready": True/False, "message": str}
    """
    try:
        import ollama as ol
        models_response = ol.list()

        # ol.list() returns an object with a 'models' attribute
        available_names = []
        if hasattr(models_response, 'models'):
            available_names = [m.model for m in models_response.models]
        elif isinstance(models_response, dict):
            available_names = [m.get("name", "") for m in models_response.get("models", [])]

        model_ready = any(model in name for name in available_names)

        if not model_ready:
            return {
                "available": True,
                "model_ready": False,
                "message": (
                    f"Ollama is running but '{model}' is not pulled.\n"
                    f"Run: ollama pull {model}\n"
                    f"Available models: {available_names or 'none'}"
                ),
            }
        return {
            "available": True,
            "model_ready": True,
            "message": f"Ollama ready — {model} loaded.",
        }

    except ImportError:
        return {
            "available": False,
            "model_ready": False,
            "message": "ollama package not installed. Run: pip install ollama",
        }
    except Exception as e:
        return {
            "available": False,
            "model_ready": False,
            "message": (
                f"Ollama not running ({type(e).__name__}).\n"
                f"Start it with: ollama serve\n"
                f"Or install from: https://ollama.com/download"
            ),
        }


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from inference import  run_detection, load_image, load_model, get_annotated_image
    from xai import generate_gradcam, get_heatmap_region, save_comparison_plot, save_overlay
    import numpy as np
    model = load_model()
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        img_array, rgb_float = load_image(image_path)
        detections = run_detection(model, image_path)
        # market_scene = build_scene_description(detections)
        annotated_rgb = get_annotated_image(model, image_path)

        if detections:
            heatmap, crop_overlay, full_overlay = generate_gradcam(
                rgb_float, detections[0]
            )
            region = get_heatmap_region(heatmap)
            save_overlay(full_overlay, "./outputs/heatmap_overlay.jpg")
            save_overlay(annotated_rgb, "./outputs/yolo_annotated.jpg")
            save_comparison_plot(
                rgb_float, heatmap, full_overlay,
                annotated_rgb, detections, "./outputs/comparison.jpg"
            )
        else:
            region = "di whole image"
            heatmap = np.zeros(rgb_float.shape[:2])

    
    else: 
        print("\n── Test: busy market scene ─────────────────────────────")
        market_scene = {
            "scene_type":    "market / street scene",
            "primary":       ["person", "car", "motorcycle"],
            "counts":        {"person": 7, "car": 3, "motorcycle": 4, "handbag": 5, "bicycle": 2},
            "total_objects": 21,
            "description": (
                "This looks like a market / street scene. "
                "I can see 7 people, 3 cars, and 4 motorcycles. "
                "There are also handbags and bicycles in the scene."
            ),
        }

    # print("── Ollama health check ─────────────────────────────────")
    # status = check_ollama()
    # print(f"  Available  : {status['available']}")
    # print(f"  Model ready: {status['model_ready']}")
    # print(f"  Message    : {status['message']}")

      
    result = build_pidgin_from_scene(detections, heatmap_region= region, force_tier="gemini")
    print(f"\n  Output:\n  {result}")

    # print("\n── Test: single object fallback ────────────────────────")
    # single = build_explanation_template("bottle", 0.82, "di middle part")
    # print(f"  Output:\n  {single}")
