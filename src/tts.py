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
    with open(config_path, "r") as f:
        raw = json.load(f)
    config = XttsConfig()
    for key, value in raw.items():
        if key == 'model_args':
            continue
        try:
            setattr(config, key, value)
        except Exception:
            pass

    if "model_args" in raw:
        valid_fields= set(XttsArgs.__dataclass_fields__.keys())
        filtered_args = {k: v for k, v in raw["model_args"].items() if k in valid_fields}
        config.model_args = XttsArgs(**filtered_args)

    
    tts_model = Xtts.init_from_config(config)
    tts_model.load_checkpoint(
        config,
        checkpoint_dir=model_dir,
        eval=True
    )

    # Use GPU if available, fall back to CPU
    # NOTE: CPU inference takes ~15-30s per sentence — acceptable for prototype
    tts_model.to("cpu")

    print(f"[TTS] Afro-TTS loaded: {type(tts_model)}")
    return tts_model, config

def eleven_labs_fallback(text:str):
    import os
    from dotenv import load_dotenv
    import uuid
    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs


    load_dotenv()
    ELEVENLABS_API_KEY= os.getenv("ELEVENLABS_API_KEY")

    client = ElevenLabs(
        api_key=ELEVENLABS_API_KEY,
    )

    response= client.text_to_speech.convert(
        voice_id="IAkWen5Y9zgtcrKepkq8",
        output_format="wav_24000",
        text=text,
        model_id="eleven_flash_v2_5",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )

    save_file_path = f"{uuid.uuid4()}.wav"
    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    print(f"{save_file_path}: A new audio file was saved successfully!")
    return save_file_path

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

def synthesise_sentences(chunk, tts_model, speaker_embedding, gpt_cond_latent) :
    """
    Synthesise a list of sentences and concatenate the resulting audio.
    This allows us to handle longer texts without overwhelming the model.
    """
    outputs = tts_model.inference(
            chunk,
            gpt_cond_latent=gpt_cond_latent,  # Pass the latent representation for context
            speaker_embedding=speaker_embedding,
            language="en",  
            enable_text_splitting=False  # Pidgin English routes through English
        )

    
    return outputs["wav"]


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
    chunks= split_sentences(text)
    
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

def facebook_fallback(text: str):
    """
    Facebook TTS fallback via Hugging Face Inference API.
    Requires an internet connection but no local model or GPU.
    Voice will be a generic English accent (not African).
    Use this if ElevenLabs API is unavailable or rate-limited.
    """
    from transformers import AutoTokenizer, VitsModel
    import torch
    import scipy

    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        speech = model(**inputs).waveform  

    # Convert to audio and save (implementation depends on model output format)
    # This is a placeholder; actual conversion code will depend on the model's output
    output_path = os.path.join(BASE_DIR, "..", "audios", f"{uuid.uuid4()}.wav")
    audio_data = speech.cpu().numpy()
    if audio_data.ndim > 1:
        audio_data= audio_data.flatten()
        
    audio_data = (audio_data * 32767).astype(np.int16)  # Convert to 16-bit PCM
    scipy.io.wavfile.write(output_path, rate= model.config.sampling_rate, data=audio_data)
    
    # audio_data = speech.cpu().numpy()
    # save_wav(audio_data, output_path=output_path)
    
    print(f"[TTS-FacebookFallback] Spoken: \"{text}\"")
    return output_path
# -------testing-------────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     # Simple test of the TTS module with a sample Pidgin English sentence.
    
#     model, config = load_afro_tts()
#     gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(
# audio_path= REFERENCE_WAV
#     )

#     sample_text = "my name is Bolu. I see people and car for di road. i am eight years old. i have a lot of homework to do."
#     chunks = split_sentences(sample_text)
#     print(f"Split into sentences: {chunks}")
#     output_wav = speak(sample_text, model, reference_wav="./models/afro-tts/audios/reference_accent3.wav")
#     _play_audio(output_wav)
    # Uncomment to test fallback TTS
    # speak_fallback(sample_text)