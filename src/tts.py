# src/tts.py
# Afro-TTS integration with Pidgin English output
#
# SETUP — run once before using this module:
#
#   pip install TTS huggingface_hub scipy
#   pip install huggingface_hub[cli]
#   huggingface-cli download intronhealth/afro-tts --local-dir ./models/afro-tts
#

import os
import numpy as np
from scipy.io.wavfile import write as write_wav
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import uuid
import torch
import torchaudio

# ── Afro-TTS loader ────────────────────────────────────────────────────────────

def load_afro_tts(model_dir: str = "./models/afro-tts"):
    """
    Load the Afro-TTS XTTS model from a local directory.
    First run will be slow (~30s). Subsequent runs use cached weights.
    """
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts, XttsArgs
    import torch
    import json

    config_path = os.path.join(model_dir, "config.json")
    print(f"--- Loading Bolu's Config from {config_path} ---")
    config = XttsConfig()
    config.load_json(config_path)
    
    # 2. Initialize model structure
    model = Xtts.init_from_config(config)
    
    # 3. Determine Device (Always use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Moving model to {device} ---")
    
    # 4. Load weights directly to the model
    # On a local GPU laptop, we point directly to the folder
    model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
    model.cuda()  # Ensure model is on GPU for inference
    
    print("--- Afro-TTS (Bolu) is ready on GPU! ---")
    return model, config


def synthesise_sentences(chunk, config, tts_model, reference_wav="./models/afro-tts/audios/reference_accent.wav"):
    """
    Synthesise a list of sentences and concatenate the resulting audio.
    This allows us to handle longer texts without overwhelming the model.
    """
    
    outputs = tts_model.synthesize(
            chunk,
            config,
            gpt_cond_len=3,  # Pass the latent representation for context
            speaker_wav=reference_wav,
            language="en",  
            enable_text_splitting=False  # Pidgin English routes through English
        )
    output_path=f"audios/{uuid.uuid4()}.wav"
    torchaudio.save(output_path, torch.tensor(outputs["wav"]).unsqueeze(0), sample_rate=24000)
    
    return output_path


def save_wav(audio_data: np.ndarray, sample_rate: int = 22050, output_path: str = "./audios/output.wav"):
     # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_wav(output_path, 24000, audio_data)
    print(f"[TTS] Audio saved to: {output_path}")

    return output_path

def _play_audio(path: str):
    """Play a .wav file — cross-platform best effort."""
    import platform
    system = platform.system()
    try:
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
    except Exception as e:
        print(f"[TTS] Autoplay failed (audio still saved): {e}")


# # ── Fallback TTS (offline, no GPU needed) ─────────────────────────────────────
BASE_DIR= os.path.dirname(os.path.abspath(__file__))
import uuid

def speak_fallback(text: str, output_path: str = os.path.join(BASE_DIR, "..", "audios", "{uuid.uuid4()}.wav")):
    """
    pyttsx3 fallback — fully offline, no GPU, no model download.
    Voice will be a generic system voice (not African accented).
    Use this during development when Afro-TTS is slow or unavailable.
    """
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 155)     # slightly slower — clearer for demos
    engine.setProperty("volume", 0.95)
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    print(f"[TTS-fallback] Spoken: \"{text}\"")

    return output_path


