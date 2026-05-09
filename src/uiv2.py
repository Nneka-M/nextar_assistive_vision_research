import gradio as gr
from tts import load_afro_tts, synthesise_sentences
from explainer import build_pidgin_from_image
from pathlib import Path
from db import log_detection, init_db
import os
import shutil
import base64

init_db()

BASE_DIR= os.path.dirname(os.path.abspath(__file__))
REFERENCE_WAV=os.path.join(BASE_DIR, "..", "models", "afro-tts", "audios", "reference_accent.wav")

print("[API] Loading Afro-TTS model and speaker embedding at startup...")
tts_model, tts_config = load_afro_tts()

print("Startup complete.")

def render_bounding_boxes(image_path: str, gemini_objects: list) -> str:
    """
    Returns an HTML string with the image and 
    bounding box overlays positioned via CSS.
    """
    # Convert image to base64 for embedding
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = image_path.split(".")[-1].lower()
    mime = "image/jpeg" if ext in ("jpg","jpeg") else "image/png"

    boxes_html = ""
    colors = ["#3B82F6","#10B981","#F59E0B","#EF4444","#8B5CF6"]
    
    for i, obj in enumerate(gemini_objects):
        y1, x1, y2, x2 = obj["box_2d"]
        color = colors[i % len(colors)]
        left   = x1 / 10      # convert 0-1000 to 0%-100%
        top    = y1 / 10
        width  = (x2 - x1) / 10
        height = (y2 - y1) / 10
        label  = obj.get("label", "object")

        boxes_html += f"""
        <div style="
            position:absolute;
            left:{left}%; top:{top}%;
            width:{width}%; height:{height}%;
            border: 2px solid {color};
            box-sizing:border-box;
            pointer-events:none;">
            <span style="
                position:absolute; top:-22px; left:0;
                background:{color}; color:white;
                font-size:11px; padding:2px 6px;
                border-radius:3px; white-space:nowrap;">
                {label}
            </span>
        </div>"""

    return f"""
    <div style="position:relative; display:inline-block; width:100%;">
        <img src="data:{mime};base64,{b64}" 
             style="width:100%; display:block;"/>
        {boxes_html}
    </div>"""

def run_pipiline(image):

    scene_description = build_pidgin_from_image(image)
    english= scene_description.get("description_english")
    print(f"[Scene Description] {english}")
    audio_path = synthesise_sentences(english, tts_model=tts_model, config=tts_config, reference_wav=REFERENCE_WAV)
    objects = scene_description.get("objects", [])  # this is what goes into render_bounding_boxes

    box_html = render_bounding_boxes(image, objects)  # Placeholder: pass actual detected objects here
    log_detection(image, 'phase2', [], "", english, audio_path)
    return box_html, english, audio_path

demo= gr.Interface(
    fn=run_pipiline,
    inputs=gr.Image(type="filepath", label ="upload or capture image"),
    outputs=[
        gr.HTML(label="Detected Objects with Bounding Boxes"),
        gr.Textbox(label="Scene Description (English)"),
        gr.Audio(label="Audio Description", autoplay=True)
    ],
    title="Nextar - Assistive Vision (Prototype)",
    description= "Upload an image to see detected objects, a scene description in  English, and hear the audio description in a Nigerian accent."
)

demo.launch(share=True)