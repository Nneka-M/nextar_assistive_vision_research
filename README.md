# Nextar - Assistive Vision Research

## Overview

**Nextar- Assistive Vision** is an AI-powered assistive vision system designed to help visually impaired individuals understand their surroundings through scene analysis and natural African accent audio descriptions. The system combines computer vision object detection, explainable AI (XAI) techniques, and text-to-speech synthesis to provide comprehensive scene descriptions in English with Nigerian English accent.

### Key Features

- **Object Detection**: Real-time detection of multiple objects in a scene using YOLO11n
- **Explainable AI**: Grad-CAM heatmaps showing which visual features drove the model's decisions
- **Natural Audio Output**: Afro-TTS synthesis with Nigerian English accent for accessibility
- **Pidgin English Generation**: LLM-powered natural language explanations in Pidgin English
- **Interactive UI**: Gradio-based interface for easy image uploads and real-time feedback
- **Logging & Analytics**: SQLite database tracking user interactions and model performance

---

## What It Does

The Nextar system implements **two distinct architectural paths** for comparative research analysis:

### **Path 1: Modular Pipeline** (Original - `ui.py`)
Offline-first approach emphasizing explainability through component isolation:
1. **Image Ingestion**: User uploads via Gradio
2. **Object Detection**: YOLO11n identifies objects with bounding boxes
3. **Visual Explanation**: Grad-CAM (MobileNetV3) generates attention heatmaps showing model reasoning
4. **Scene Understanding**: Gemini API or Claude API generates English description
5. **Audio Synthesis**: Afro-TTS converts description to speech with Nigerian English accent
6. **Logging**: Interactions recorded with `phase1` tag in SQLite

**Output**: Annotated image, attention heatmap, text description, audio narration

**Strengths**: Privacy (offline), interpretable attention maps, modular debugging
**Challenges**: Latency (~5-12s), LLM vision understanding limited to text descriptions

---

### **Path 2: Multimodal End-to-End** (New - `uiv2.py`)
Unified API-first approach leveraging vision-language models:
1. **Image Ingestion**: User uploads via Gradio
2. **Unified Analysis**: Google Gemini multimodal model analyzes image directly
3. **Dynamic Bounding Box Generation**: Extracts object coordinates from Gemini response
4. **Scene Understanding**: Gemini generates complete description and bounding boxes in single inference
5. **HTML Rendering**: CSS-based bounding box visualization overlaid on image
6. **Audio Synthesis**: Afro-TTS converts description to speech
7. **Logging**: Interactions recorded with `phase2` tag in SQLite

**Output**: HTML interactive visualization with bounding boxes, text description, audio narration

**Strengths**: End-to-end multimodal understanding, faster inference, native bounding box generation
**Tradeoffs**: Requires API access (not offline), lacks explicit attention mechanisms

---

## Infrastructure & Architecture

### **Path 1: Modular Explainable Pipeline** (`ui.py`)

#### 1. **Object Detection** (`src/inference.py`)
- **Model**: YOLO11n (nano version optimized for CPU)
- **Performance**: ~100-200ms per image
- **Outputs**: Bounding boxes, class labels, confidence scores

#### 2. **Explainable AI** (`src/xai.py`)
- **Method**: Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Model**: MobileNetV3-Large (lightweight, CPU-friendly)
- **Purpose**: Generate attention heatmaps showing visual focus regions
- **Performance**: ~800ms-1.5s per image
- **Outputs**: Attention heatmap, region descriptions, overlay visualization

#### 3. **Natural Language Generation** (`src/explainer.py`)
- **Tier 1**: Gemini(local, offline, ~2-5s)
- **Tier 2**: Claude API via Anthropic (fallback)
- **Output**:  English description based on detected objects + attention regions

#### 4. **Text-to-Speech** (`src/tts.py`)
- **Model**: Afro-TTS (XTTS-based, Nigerian English accent)
- **Performance**: ~2-5s per sentence

---

### **Path 2: Unified Multimodal Architecture** (`uiv2.py`)

#### 1. **Unified Vision-Language Analysis** (`src/explainer.py` - Gemini mode)
- **Model**: Google Gemini Pro Vision (multimodal)
- **Architecture**: Single inference endpoint handles detection, localization, and description
- **Inputs**: Raw image
- **Outputs**: 
  - Object list with 2D bounding box coordinates (normalized 0-1000 scale)
  - Scene description in  English
  - All analysis in one API call (~2-5s)

#### 2. **Dynamic Bounding Box Rendering** (HTML/CSS in `uiv2.py`)
- **Method**: `render_bounding_boxes()` function creates HTML overlay
- **Format**: Base64-encoded image with CSS-positioned boxes
- **Visualization**: Interactive bounding boxes with labels
- **No separate XAI model needed**: Coordinates extracted directly from Gemini response

#### 3. **Text-to-Speech** (`src/tts.py`)
- **Same as Path 1**: Afro-TTS with Nigerian English accent
- **Performance**: ~2-5s per sentence

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
- **LLM**: Gemini API , Anthropic API (optional)
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

### **Path 1: Modular Explainable Pipeline** (Original Architecture)
```bash
python src/ui.py
```
- Opens at `http://localhost:7860`
- Offline-first with explicit attention visualization
- Full visibility into each processing step
- Database logs as `phase1`

### **Path 2: Unified Multimodal Architecture** (New Architecture)
```bash
python src/uiv2.py
```
- Opens at `http://localhost:7860`
- Requires Google Gemini API key (`GOOGLE_API_KEY` environment variable)
- Single unified analysis with dynamic bounding boxes
- Database logs as `phase2`
- **Note**: Set `GOOGLE_API_KEY` before running

---

### FastAPI Backend
```bash
python -m uvicorn src.api:app --reload --port 8000
```
- Swagger UI: `http://localhost:8000/docs`
- Supports both Path 1 and Path 2 endpoints

### Test Script
```bash
python main.py
```

---

## Performance & Comparative Analysis

### **Path 1: Modular Pipeline**

| Component | Model | Latency (CPU) |
|-----------|-------|---------------|
| Object Detection | YOLO11n | 100-200ms |
| Grad-CAM XAI | MobileNetV3 | 800-1500ms |
| LLM (Pidgin Gen) | GEmini | 2-5s |
| TTS | Afro-TTS XTTS | 2-5s per sentence |
| **Total Pipeline** | - | **5-12s** |

**Advantages**:
- Fully offline-first (privacy-preserving)
- Explicit attention visualization for interpretability
- Modular debugging and component replacement
- Predictable, controlled inference pipeline

**Limitations**:
- Higher latency due to sequential processing
- LLM has limited vision understanding (text-only input)
- Requires local model downloads and resources

---

### **Path 2: Unified Multimodal**

| Component | Model | Latency |
|-----------|-------|---------|
| Unified Analysis | Google Gemini Pro Vision | 2-5s |
| Bounding Box Rendering | HTML/CSS | <100ms |
| TTS | Afro-TTS XTTS | 2-5s per sentence |
| **Total Pipeline** | - | **4-10s** |

**Advantages**:
- Lower latency (single API call vs. multi-stage pipeline)
- Native multimodal understanding (vision + language jointly)
- Automatic bounding box generation
- Simpler architecture, fewer components

**Limitations**:
- Requires API key (not offline)
- Lacks explicit attention visualization
- API dependency and cost considerations
- Less component-level interpretability

---

### **Research Value**

This dual-path implementation enables comparative research on:
1. **Latency vs. Interpretability**: Offline modular approach vs. unified multimodal speed
2. **Explainability Trade-offs**: Explicit Grad-CAM heatmaps vs. implicit multimodal reasoning
3. **Accuracy Assessment**: YOLO+LLM vs. Gemini's native vision understanding
4. **Accessibility Trade-offs**: Privacy (offline) vs. performance (cloud API)
5. **Resource Requirements**: CPU-only vs. API-dependent architectures

---

## Research Applications

This system supports two distinct research paradigms:

**Path 1 - Modular Explainability Research**:
- Accessibility evaluation with local inference (privacy-first)
- XAI methodology testing (Grad-CAM attention analysis)
- Component-level bias detection (object detection vs. explanation generation)
- Resource-constrained deployment scenarios
- Offline system reliability

**Path 2 - End-to-End Multimodal Research**:
- Vision-language model performance in accessibility contexts
- API-dependent assistive system design trade-offs
- Unified multimodal understanding vs. modular approaches
- Real-world deployment feasibility
- Cost-performance analysis for cloud-based assistive AI

**Comparative Research Goals**:
- Latency vs. interpretability trade-offs in assistive AI
- Privacy (offline) vs. performance (cloud APIs) for visually impaired users
- Explicit XAI (Grad-CAM) vs. implicit multimodal reasoning
- Modular debugging capabilities vs. end-to-end robustness
- Multilingual TTS effectiveness with different vision architectures

---

## Key Design Decisions

**Path 1 (Modular Pipeline)**:
1. **YOLO11n over larger models**: CPU efficiency without sacrificing scene description accuracy
2. **Separate MobileNetV3 for XAI**: Clean layer attribution for attention visualization
3. **Tiered LLM approach**: Offline Ollama primary, Claude fallback for quality comparison
4. **Pidgin English focus**: Natural language for diverse African audiences
5. **Offline-first**: Privacy and reliability for assistive AI

**Path 2 (Unified Multimodal)**:
1. **Gemini multimodal**: Native vision-language understanding in single inference
2. **API-first architecture**: Leverage latest models without local deployment
3. **Dynamic bounding box extraction**: Reduce complexity vs. separate detection
4. **HTML/CSS rendering**: Interactive visualization without additional ML models
5. **Comparative research**: Demonstrate trade-offs between architectures

**Unified Design**:
- **SQLite logging with phase tagging**: Track both architectures in shared database
- **Shared TTS pipeline**: Consistent audio output across both paths
- **Modular Gradio interfaces**: Independent UIs for each architecture (`ui.py` vs. `uiv2.py`)

---

## Future Enhancements

**Path 1 Improvements**:
- GPU acceleration for detection and XAI
- Real-time video processing
- Fine-tuned local LLM for better Pidgin generation
- Advanced XAI methods beyond Grad-CAM (LIME, SHAP)

**Path 2 Improvements**:
- Model comparison (Gemini vs. other multimodal models)
- Cost optimization and caching strategies
- Offline fallback mechanisms
- Custom bounding box refinement

**Cross-Path Research**:
- User study comparing explanatory quality (heatmaps vs. multimodal)
- Latency vs. accuracy trade-off analysis
- Privacy vs. performance user preferences
- Accessibility effectiveness measurement
- Mobile deployment of both architectures

---

## License & Credits

- **YOLO**: Ultralytics (AGPL-3.0)
- **Afro-TTS**: Intron Health
- **Grad-CAM**: Selvaraju et al.
- **Research Focus**: Assistive AI & Explainability for Disability

---

## Contact & Contributions

For questions, issues, or contributions, please refer to project documentation or contact the research team.

