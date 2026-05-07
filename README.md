# Nextar - Assistive Vision Research

## Overview

**Nextar** is an AI-powered assistive vision system designed to help visually impaired individuals understand their surroundings through real-time scene analysis and natural audio descriptions. The system combines computer vision object detection, explainable AI (XAI) techniques, and text-to-speech synthesis to provide comprehensive scene descriptions in Pidgin English with Nigerian English accent.

### Key Features

- **Object Detection**: Real-time detection of multiple objects in a scene using YOLO11n
- **Explainable AI**: Grad-CAM heatmaps showing which visual features drove the model's decisions
- **Natural Audio Output**: Afro-TTS synthesis with Nigerian English accent for accessibility
- **Pidgin English Generation**: LLM-powered natural language explanations in Pidgin English
- **Interactive UI**: Gradio-based interface for easy image uploads and real-time feedback
- **Logging & Analytics**: SQLite database tracking user interactions and model performance

---

## What It Does

The Nextar system processes images in a multi-stage pipeline:

1. **Image Ingestion**: User uploads an image via the Gradio UI
2. **Object Detection**: YOLO11n identifies all objects in the scene with bounding boxes
3. **Visual Explanation**: Grad-CAM generates attention heatmaps highlighting model reasoning
4. **Scene Understanding**: LLM  generates a natural Pidgin English description based on detected objects and attention regions
5. **Audio Synthesis**: Afro-TTS converts the description to speech with Nigerian English accent
6. **Logging**: All interactions (detections, descriptions, audio) are recorded to SQLite database

**Output**: User receives:
- Annotated image with detected objects
- Attention heatmap showing focus areas
- Scene description in Pidgin English (text)
- Audio narration of the scene

---

## Infrastructure & Architecture

### Core Components

#### 1. **Object Detection** (`src/inference.py`)
- **Model**: YOLO11n (nano version optimized for CPU)
- **Why YOLO11n**: 3-4x faster than YOLO11s with acceptable accuracy for scene description
- **Performance**: ~100-200ms per image on modern CPU
- **Inputs**: Image file path or PIL Image
- **Outputs**: Bounding boxes, class labels, confidence scores

#### 2. **Explainable AI** (`src/xai.py`)
- **Primary XAI Method**: Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Model**: MobileNetV3-Large (lightweight, CPU-friendly)
- **Why Separate Model**: YOLO's multi-scale detection head is not amenable to layer attribution. Instead, we crop the primary detected object and run MobileNetV3 on that crop for focused explanation.
- **Performance**: ~800ms-1.5s per image on CPU
- **Outputs**: 
  - Heatmap visualization (attention map)
  - Region descriptions (center, edges, etc.)
  - Overlay visualization

#### 3. **Natural Language Generation** (`src/explainer.py`)
- **Tier 1 (Primary)**: Ollama + Gemma 3:1B (local, offline, ~2-5s)
- **Tier 2 (Fallback)**: Claude API via Anthropic (if API key provided)
- **Tier 3 (Minimum)**: Hardcoded fallback string
- **Output Language**: Pidgin English with natural, context-aware phrasing
- **Process**: Converts detected objects + heatmap region into natural description via LLM prompt

#### 4. **Text-to-Speech** (`src/tts.py`)
- **Model**: Afro-TTS (XTTS-based, trained on African languages)
- **Voice**: Nigerian English accent with pre-recorded reference audio
- **Performance**: ~2-5s per sentence on CPU
- **Output Format**: 16kHz WAV files
- **Features**: Batch processing support, concurrent synthesis

#### 5. **Web API** (`src/api.py`)
- **Framework**: FastAPI with CORS middleware
- **Endpoints**:
  - `POST /speak` - TTS synthesis endpoint
  - Additional endpoints for detection and explanation
- **Static Files**: Serves generated results via `/outputs` route
- **Lifespan Management**: Loads TTS model at startup for performance

#### 6. **User Interface** (`src/ui.py`)
- **Framework**: Gradio
- **Title**: "Nextar - Assistive Vision (Prototype)"
- **Inputs**: Image upload/capture interface
- **Outputs**:
  - Detected Objects (annotated image)
  - Attention Heatmap (Grad-CAM visualization)
  - Scene Description (Pidgin English text)
  - Audio Description (auto-playing audio)
- **Backend**: Runs complete pipeline in sequence

#### 7. **Data Logging** (`src/db.py`)
- **Database**: SQLite at `./data/nextar.db`
- **Table**: `interactions` table with columns:
  - `id`, `timestamp`, `image_path`, `detections`, `heatmap_region`, `scene_description`, `audio_path`
- **Purpose**: Track all user interactions for research analysis

---

## Project Directory Structure

```
nextar_assistive_vision_research/
├── README.md                          # This file
├── pyproject.toml                     # Python project metadata & dependencies
├── main.py                            # Entry point (placeholder)
├── download_model.py                  # Script to download pre-trained models
│
├── src/                               # Core source code
│   ├── inference.py                   # YOLO11n object detection
│   ├── xai.py                         # Grad-CAM heatmap generation
│   ├── explainer.py                   # LLM-powered Pidgin English generation
│   ├── tts.py                         # Afro-TTS speech synthesis
│   ├── api.py                         # FastAPI web service
│   ├── ui.py                          # Gradio user interface
│   ├── db.py                          # SQLite database management
│   └── __pycache__/                   # Python bytecode cache
│
├── models/                            # Pre-trained model weights
│   └── afro-tts/                      # Afro-TTS XTTS model
│       ├── config.json                # Model configuration
│       ├── model.pth                  # Model weights
│       ├── dvae.pth                   # Variational autoencoder
│       ├── mel_stats.pth              # Mel-spectrogram statistics
│       ├── vocab.json                 # Vocabulary file
│       ├── LICENSE.txt                # Afro-TTS license
│       ├── README.md                  # Afro-TTS documentation
│       └── audios/                    # Reference audio samples
│           └── reference_accent.wav   # Nigerian English reference
│
├── data/                              # Data storage
│   └── nextar.db                      # SQLite database (auto-created)
│
├── test_images/                       # Sample test images
├── temp_uploads/                      # Temporary file storage
├── audios/                            # Audio samples and outputs
├── results_folder/                    # API output results
├── yolo11n.pt                         # YOLO11n model weights (~5 MB)
└── new/                               # Miscellaneous files
```

---

## Dependencies & Setup

### Python Version
- **Required**: Python ≥ 3.10

### Key Dependencies
- **Computer Vision**: `ultralytics` (YOLO), `opencv-python`, `torchvision`
- **XAI**: `grad-cam`, `torch`
- **TTS**: `coqui-tts` (Afro-TTS), `torchaudio`
- **LLM**: `ollama` (local), Anthropic API (optional)
- **Web/UI**: `fastapi`, `gradio`
- **Audio**: `scipy`
- **ML/Data**: `numpy`, `pandas`, `transformers`, `pillow`

See `pyproject.toml` for complete dependency list and versions.

### Installation Steps

1. **Clone/setup the project**:
   ```bash
   cd nextar_assistive_vision_research
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .  # Or: pip install -r requirements.txt
   ```

4. **Download Afro-TTS model** (if not already in `models/afro-tts/`):
   ```bash
   huggingface-cli download intronhealth/afro-tts --local-dir ./models/afro-tts
   ```

5. **Setup Ollama** (for local LLM):
   - Download from [ollama.com](https://ollama.com/download)
   - Pull the model: `ollama pull gemma3:1b`
   - Verify it's running: `ollama list`

6. **Optional: Setup Anthropic API**:
   ```bash
   export ANTHROPIC_API_KEY="sk-..."  # Or add to .env file
   ```

---

## Running the Application

### Gradio UI (Recommended)
```bash
python src/ui.py
```
- Opens at `http://localhost:7860`
- Drag & drop or upload an image
- See real-time detection, heatmap, Pidgin description, and audio output

### FastAPI Backend
```bash
python -m uvicorn src.api:app --reload --port 8000
```
- Swagger UI: `http://localhost:8000/docs`
- Available endpoints: `/speak`, `/detect`, `/explain`, etc.

### Test Script
```bash
python main.py
```

---

## Performance Metrics

| Component | Model | Latency (CPU) | Hardware Assumed |
|-----------|-------|---------------|------------------|
| Object Detection | YOLO11n | 100-200ms | Modern CPU |
| Grad-CAM | MobileNetV3 | 800-1500ms | Modern CPU |
| LLM (Pidgin Gen) | Ollama + Gemma 3:1B | 2-5s | Modern CPU |
| TTS | Afro-TTS XTTS | 2-5s per sentence | Modern CPU |
| **Total Pipeline** | - | ~5-12s | Modern CPU |

---

## Research Applications

This system is designed for:
- **Accessibility Research**: Evaluating assistive vision quality for visually impaired users
- **XAI in Practice**: Real-world testing of Grad-CAM explanations
- **Multilingual TTS**: African language synthesis research
- **Human-Computer Interaction**: User studies on scene understanding
- **AI Ethics**: Bias detection in object detection and explanation generation

---

## Key Design Decisions

1. **YOLO11n over larger models**: Prioritizes CPU efficiency without sacrificing accuracy for scene description
2. **Separate MobileNetV3 for XAI**: YOLO's architecture is incompatible with layer attribution; cropping + MobileNetV3 provides clean explanations
3. **Tiered LLM approach**: Offline Ollama for privacy/speed, with Claude fallback for better quality
4. **Pidgin English focus**: Natural language for diverse African audiences with linguistic authenticity
5. **SQLite logging**: Lightweight, portable database for research data collection

---

## Future Enhancements

- Real-time video processing
- Multi-modal input (camera feed, audio input)
- Fine-tuned LLM for better Pidgin generation
- GPU acceleration options
- Mobile deployment (Flutter/React Native UI)
- User feedback loop for model refinement

---

## License & Credits

- **YOLO**: Ultralytics (AGPL-3.0)
- **Afro-TTS**: Intron Health
- **Grad-CAM**: Selvaraju et al.
- **Research Focus**: Assistive AI & Explainability for Disability

---

## Contact & Contributions

For questions, issues, or contributions, please refer to project documentation or contact the research team.

