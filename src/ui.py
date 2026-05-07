import gradio as gr
from inference import load_model, load_image, run_detection, get_annotated_image
from tts import load_afro_tts, speak, save_wav, synthesise_sentences
from xai import generate_gradcam,  get_heatmap_region
from explainer import build_pidgin_from_scene
from pathlib import Path
from db import log_detection, init_db
import os
import shutil

init_db()
yolo_model = load_model()

BASE_DIR= os.path.dirname(os.path.abspath(__file__))
REFERENCE_WAV=os.path.join(BASE_DIR, "..", "models", "afro-tts", "audios", "reference_accent.wav")

# tts_model, tts_config, speaker_embedding, gpt_cond_latent
print("[API] Loading Afro-TTS model and speaker embedding at startup...")
tts_model, tts_config = load_afro_tts()
# gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(
# audio_path= REFERENCE_WAV
#     )
print("Startup complete.")

def run_pipiline(image):
    # os.makedirs("temp_uploads", exist_ok=True)
    # save_path= f".temp_uploads/{os.path.basename(image)}"
    # shutil.copy(image, save_path)


    img_array, rgb_float = load_image(image)
    detections = run_detection(yolo_model, image)
    annotated_image = get_annotated_image(yolo_model, image)
    heatmap, _, full_overlay = generate_gradcam(rgb_float, detections)
    region= get_heatmap_region(heatmap)
    scene_description = build_pidgin_from_scene(detections, region, force_tier= "gemini")
    audio_path = synthesise_sentences(scene_description, tts_model=tts_model, config=tts_config, reference_wav=REFERENCE_WAV)
    log_detection(image, detections, region, scene_description, audio_path)
    return annotated_image, full_overlay, scene_description, audio_path

demo= gr.Interface(
    fn=run_pipiline,
    inputs=gr.Image(type="filepath", label ="upload or capture image"),
    outputs=[
        gr.Image(label="Detected Objects"),
        gr.Image( label="Attention heatmap"),
        gr.Textbox(label="Scene Description (Pidgin English)"),
        gr.Audio(label="Audio Description", autoplay=True)
    ],
    title="Nextar - Assistive Vision (Prototype)",
    description= "Upload an image to see detected objects, attention heatmap, a scene description in Pidgin English, and hear the audio description."
)

demo.launch(share=True)