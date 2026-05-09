# src/explainer.py  (v3 — LLM-powered, no hardcoded sentence structure)
#
# Uses a LLM  to generate natural  English explanations.
# No hardcoded templates, no lookup tables, no rigid sentence patterns.
#
# 
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
import json


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
- Do not exceed 40 words. Be concise and clear.
"""

    return prompt


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

    
    # ── Tier 2: Gemini API ──
    if force_tier in (None, "gemini"):
        # if api_key:
            try:
                result = _call_gemini(prompt)
                print("[Explainer] Used gemini API.")
                return result
            except Exception as e:
                print(f"[Explainer] gemini API failed ({type(e).__name__}) — using fallback.")
        
    # ── Tier 3: Bare fallback ──
    print("[Explainer] Using groq fallback (gemini not available).")
    return _call_groq(prompt)

import re

def build_pidgin_from_image(image_path: str) -> str:
    import base64
    """Phase 2 — pass raw image directly to Gemini Vision."""
    from google import genai
    from google.genai import types
    load_dotenv()

    with open(image_path, "rb") as f:
        image_bytes = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.split(".")[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = """You are helping a visually impaired person understand the scene/setting in front of them.

Analyse this image carefully and write 2-3 sentences. Do not exceed 250 characters total, describing the scene in front of you. Focus on the most important and relevant details that would help a visually impaired person understand what is in the image. Use the following structure:
1. First in natural English describing the scene — what objects are present, 
   where they are, what activity or context is implied.

Rules:
- Write only the sentences which must be under  250 characters total, nothing else"
- No preamble, no quotation marks
- Sound like a friendly helpful person talking naturally
- return a JSON object with exactly this structure:
{
  "description_english": "...",
  "objects": [
    {
      "label": "person",
      "confidence": "high",
      "box_2d": [y_min, x_min, y_max, x_max]
    }
  ]
}

Bounding box coordinates are normalised 0-1000 (Gemini's format).
Return only valid JSON, no markdown fences, no preamble."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=base64.b64decode(image_bytes), mime_type=mime),
            types.Part.from_text(text=prompt)
        ],
    )

    raw = response.text
    
    try:
        start_index = raw.find('{')
    
        if start_index != -1:
            json_part = raw[start_index:]
            parsed = json.loads(json_part)
            print(parsed)
    except json.JSONDecodeError:
        # Graceful fallback if Gemini doesn't return clean JSON
        parsed = {
            "description_english": raw,
            "objects": []
        }
    return parsed


