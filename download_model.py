# from huggingface_hub import snapshot_download
# import os

# # Define where you want the model to live on your laptop
# target_folder = r"C:\Users\USER\Documents\nextar_assistive_vision_research\models\afro-tts"

# print(f"Downloading Afro-TTS to {target_folder}...")

# snapshot_download(
#     repo_id="intronhealth/afro-tts",
#     local_dir=target_folder,
#     local_dir_use_symlinks=False  # This makes it a real folder with real files
# )

# print("Download Complete!")

import torch
import torchaudio

print(f"Torch: {torch.__version__}")
print(f"Audio: {torchaudio.__version__}")
print(f"CUDA status: {torch.cuda.is_available()}")