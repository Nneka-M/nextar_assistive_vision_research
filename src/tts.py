# src/tts.py
# Afro-TTS integration with Pidgin English output
#
# SETUP — run once before using this module:
#
#   pip install TTS huggingface_hub scipy
#   pip install huggingface_hub[cli]
#   huggingface-cli download intronhealth/afro-tts --local-dir ./models/afro-tts
#
# You also need a ~6 second Nigerian English reference audio clip.
# Option A: record yourself or a colleague speaking any sentence in Nigerian English.
# Option B: download a free sample from:
#   https://commonvoice.mozilla.org/en/datasets  (filter: Nigeria)
# Save it as: audios/reference_accent.wav

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


def split_sentences(text: str, max_chars: int = 100) -> list[str]:
    """
    Split text into sentences using punctuation as delimiters.
    This helps Afro-TTS handle longer texts by processing one sentence at a time.
    """
    # Simple regex to split on ., !, ? followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]  # Remove empty strings
    # chunks = []
    # for sentence in sentences:
    #     # If sentence is still too long, split on commas
    #     if len(sentence) > max_chars:
    #         sub = [s.strip() for s in sentence.split(',') if s.strip()]
    #         # If still too long, hard split by word count
    #         for s in sub:
    #             if len(s) > max_chars:
    #                 words = s.split()
    #                 while words:
    #                     chunks.append(' '.join(words[:20]))
    #                     words = words[20:]
    #             else:
    #                 chunks.append(s)
    #     else:
    #         chunks.append(sentence)
    
    # return [c for c in chunks if c]
    print(sentences)
    return sentences

def synthesise_sentences(chunk, config, tts_model, reference_wav="./models/afro-tts/audios/reference_accent.wav"):
    """
    Synthesise a list of sentences and concatenate the resulting audio.
    This allows us to handle longer texts without overwhelming the model.
    """
    # if gpt_cond_latent.ndim == 3 and gpt_cond_latent.size(1) > 1:
    # # Average the 32 tokens into 1 single summary token
    # # Resulting shape: [1, 1, 1024]
    #     gpt_cond_latent = gpt_cond_latent.mean(dim=1, keepdim=True)
    #     print(f"Averaged GPT conditioning latent to shape: {gpt_cond_latent.shape}")

    # if speaker_embedding.shape[-1] == 1024:
    # # If it's 1024 but model wants 512, we mean pool it 
    # # and then slice it or project it. 
    # # For Afro-TTS, usually squeezing it and taking the first 512 
    # # or mean pooling works best.
    #     speaker_embedding = speaker_embedding.mean(dim=1) # Results in [1, 1024]
    
    # Most XTTS models use a 512-dim speaker embedding. 
    # If the model crashes saying it wants 512, use this line:
    # speaker_embedding = speaker_embedding[:, :512]
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



def speak(
    text: str,
    tts_model,
    speaker_embedding,
    gpt_cond_latent,
    output_path: str = "./audios/output.wav",
    autoplay: bool = True
):
    """
    Synthesise text to speech using Afro-TTS with a Nigerian English accent.

    Args:
        text:          The Pidgin English explanation string to speak.
        model:         Loaded Xtts model instance.
        config:        Loaded XttsConfig instance.
        speaker_embedding: The speaker embedding for the Nigerian English accent.
        gpt_cond_latent: The GPT conditioning latent for context.
        output_path:   Where to save the output .wav file.
        autoplay:      If True, plays audio immediately after synthesis.

    Returns:
        np.concatenate(wavs): The raw audio data as a NumPy array.
    """

    print(f"[TTS] Synthesising: \"{text[:60]}...\"" if len(text) > 60 else f"[TTS] Synthesising: \"{text}\"")
    # chunks= split_sentences(text)
    
    synthesise_fn = partial(
        synthesise_sentences,
        tts_model=tts_model,
        speaker_embedding=speaker_embedding,
        gpt_cond_latent=gpt_cond_latent,
    )
    
    with ThreadPoolExecutor(max_workers=2) as pool:
        wavs = list(pool.map(synthesise_fn, chunks))
    
    return np.concatenate(wavs)

   


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


# -------testing-------────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simple test of the TTS module with a sample Pidgin English sentence.
    REFERENCE_WAV='./models/afro-tts/audios/reference_accent.wav'
    tts_model, config = load_afro_tts()
#     gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(
# audio_path= REFERENCE_WAV
#     )
#     gpt_cond_latents = gpt_cond_latent.cuda()
#     speaker_embedding = speaker_embedding.cuda()
#     print(gpt_cond_latent.shape)
#     print(speaker_embedding.shape)

    # if speaker_embedding.ndim > 2:
    #     speaker_embedding = speaker_embedding.squeeze()
    #     if speaker_embedding.ndim == 1:
    #         speaker_embedding = speaker_embedding.unsqueeze(0)
   
    sample_text = "my name is Bolu. I see people and car for di road. i am eight years old. i have a lot of homework to do."
    # chunks = split_sentences(sample_text)
    # print(f"Split into sentences: {chunks}")
    output_wav = synthesise_sentences(sample_text, config, tts_model, reference_wav=REFERENCE_WAV )
    _play_audio(output_wav)
    # Uncomment to test fallback TTS
    # speak_fallback(sample_text)