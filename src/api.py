# api that connects TTS and XAI components, providing a simple interface for the main app

from tts import load_afro_tts, speak, split_sentences, save_wav
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from explainer import _build_prompt, _call_gemini, build_pidgin_from_scene
from inference import  run_detection, load_image, load_model, get_annotated_image
from xai import generate_gradcam, get_heatmap_region, save_comparison_plot, save_overlay
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Dict
from fastapi.staticfiles import StaticFiles
import os
from contextlib import asynccontextmanager


tts_model = None
tts_config = None
speaker_embedding = None
gpt_cond_latents = None


async def lifespan(app: FastAPI):
    global tts_model, tts_config, speaker_embedding, gpt_cond_latents
    print("[API] Loading Afro-TTS model and speaker embedding at startup...")
    tts_model, tts_config = load_afro_tts()
    gpt_cond_latents, speaker_embedding = tts_model.get_conditioning_latents(
        audio_path= "./models/afro-tts/audios/reference_accent3.wav"
    )
    print("[API] Startup complete.")
    yield

OUTPUT_DIR = "results_folder"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
app = FastAPI(name="XAI Disability Research API", lifespan = lifespan, version="1.0", description="API for TTS and XAI components in the research project.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")



    

# ── TTS Endpoint ───────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str


@app.post("/speak", description="Synthesize text to speech using Afro-TTS with a Nigerian English accent. Expects a JSON payload with a 'text' field containing the Pidgin English explanation string to speak. Returns the path to the saved .wav file.")
async def tts_endpoint(request: TTSRequest):
    """
    Endpoint to synthesise text to speech using Afro-TTS with a Nigerian English accent.
    Expects a JSON payload with a 'text' field containing the Pidgin English explanation string to speak.
    Returns the path to the saved .wav file.
    """
    model, config = load_afro_tts()

    output_wav = speak(
        text=request.text,
        model=model,
        config=config,
        speaker_embedding=speaker_embedding,
        gpt_cond_latent=gpt_cond_latents
    )

    saved_path = save_wav(output_wav)
    
    return FileResponse(
        saved_path,
        media_type="audio/wav",
    )

# ── XAI Endpoint ───────────────────────────────────────────────────────────────


class XAIResponse(BaseModel):
    scene_description: str
    gradcam_path: str
    overlay_path: str
    comparison_plot_path: str
    heatmap_region:str

@app.post("/explain", response_model=XAIResponse, description="Generate an XAI explanation for a given image. Expects a JSON payload with an 'image_path' field containing the path to the input image. Returns a JSON object with the scene description and paths to generated visualisations.")
async def xai_endpoint(File: UploadFile) -> XAIResponse:
    """
    Endpoint to generate an XAI explanation for a given image.
    Expects a JSON payload with an 'image_path' field containing the path to the input image.
    Returns a JSON object with the scene description and paths to generated visualisations.
    """
    model = load_model()
    os.makedirs("temp_uploads", exist_ok=True)
    with open(f"./temp_uploads/{File.filename}", "wb") as buffer:
        content = await File.read()
        buffer.write(content)
    image_location = f"./temp_uploads/{File.filename}"    
    img_array, rgb_float = load_image(image_location)
    detections = run_detection(model, image_location)
    annotated_rgb = get_annotated_image(model, image_location)

    # Generate Grad-CAM and visualisations
    heatmap, crop_overlay, full_overlay = generate_gradcam(rgb_float, detections)
    heatmap_region = get_heatmap_region(heatmap)
    save_overlay(full_overlay, "./outputs/heatmap_overlay.jpg")
    save_overlay(annotated_rgb, "./outputs/yolo_annotated.jpg")
    save_comparison_plot(
                rgb_float, heatmap, full_overlay,
                annotated_rgb, detections, "./outputs/comparison.jpg"
            )

    scene_description = build_pidgin_from_scene(detections, heatmap_region=heatmap_region, force_tier= "gemma")
    base_url = "http://127.0.0.1:8000/visualize"
    return {
        "scene_description": scene_description,
        "heatmap_path": f"{base_url}/{annotated_rgb}",
        "overlay_path": f"{base_url}/{full_overlay}",
        "comparison_plot_path": f"{base_url}/comparison.jpg",
        "heatmap_region": heatmap_region
    } 

@app.get("/visualize/{file_name}", description="Endpoint to visualize the annotated image. Expects a query parameter 'image_path' with the path to the annotated image. Returns the annotated image as a response.")
async def visualize(file_name: str):
    image_path = os.path.join("results_folder", file_name)
    return FileResponse(
        image_path,
        media_type="image/jpeg"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
