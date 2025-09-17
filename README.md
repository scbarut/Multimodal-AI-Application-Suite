# 🧠 Multimodal AI Application Suite

A comprehensive **multimodal AI application** suite that combines state-of-the-art vision, audio, and language models to provide a wide range of AI-powered functionalities through an intuitive web interface.

## ✨ Features

### 🖼️ Vision & Image Processing
- **Object Detection & Cut-Paste** - Detect and manipulate objects between images
- **Image-Text Similarity** - CLIP-powered semantic matching
- **Image Captioning** - Automatic image description generation
- **Optical Character Recognition (OCR)** - Extract text from images
- **Style Transfer** - Apply artistic styles between images

### 🎥 Video Analysis
- **Frame Extraction** - Extract and analyze video frames
- **Video Content Analysis** - CLIP-based video understanding

### 🔊 Audio Processing
- **Speech-to-Text** - Whisper-powered audio transcription
- **Music Genre Classification** - Automatic music genre detection

### 💬 Language & Generation
- **Visual Question Answering (VQA)** - Ask questions about images using Gemini
- **Text-to-Image Generation** - Create images from text descriptions

## 🛠️ Technology Stack

- **Frontend**: Gradio Web UI
- **Backend**: Python with various AI frameworks
- **Vision Models**: CLIP, SAM, BLIP, GroundingDINO
- **Language Models**: Gemini, Whisper
- **Generation**: Stable Diffusion, Magenta Style Transfer

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended, 8GB+ VRAM)
- Virtual environment tool

### Installation

1. **Clone the repository**

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   
   # Linux/macOS
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required models**
   
   Download the Segment Anything Model checkpoint:
   - File: `sam_vit_h_4b8939.pth`

5. **Launch the application**
   ```bash
   python main.py
   ```

6. **Access the interface**
   
   Open your browser and navigate to: `http://127.0.0.1:7860/`

## 📦 Models Overview

### 🔲 Vision Models

| Feature | Model | Purpose |
|---------|--------|---------|
| **CLIP** | `openai/clip-vit-base-patch32` | Image-text similarity |
| **Segment Anything** | `sam_vit_h_4b8939.pth` | Object segmentation |
| **Object Detection** | `IDEA-Research/grounding-dino-base` | Zero-shot detection |
| **Image Segmentation** | `CIDAS/clipseg-rd64` | Semantic segmentation |
| **Image Captioning** | `Salesforce/blip-image-captioning-base` | Auto-captioning |
| **OCR** | `easyocr` | Text extraction |
| **Style Transfer** | `magenta/arbitrary-image-stylization-v1-256/2` | Artistic style transfer |

### 🗣️ Audio & Language Models

| Feature | Model | Purpose |
|---------|--------|---------|
| **Speech-to-Text** | `openai/whisper-base` | Audio transcription |
| **Music Classification** | `dima806/music_genres_classification` | Genre detection |
| **Visual Q&A** | `gemini-1.5-flash` | Image question answering |
| **Text-to-Image** | `CompVis/stable-diffusion-v1-4` | Image generation |


## ⚙️ Configuration

### API Keys
For Gemini integration, you'll need:
- Google Generative AI API access
- Appropriate API key configuration

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only processing
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM
- **Optimal**: 32GB RAM, RTX 3080/4080 or better

## 🔧 Usage Examples

### Image-Text Similarity
Upload an image and enter text to find semantic similarity scores using CLIP.

### Object Detection & Segmentation
Use SAM and GroundingDINO to detect and segment specific objects in images.

### Visual Question Answering
Ask natural language questions about uploaded images using Gemini.

### Style Transfer
Apply artistic styles from one image to another using Magenta models.



**Note**: This application combines multiple state-of-the-art AI models and may require significant computational resources for optimal performance.