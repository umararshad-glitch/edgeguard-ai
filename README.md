# EdgeGuard AI

Offline Audio Deepfake Detection System

## Features
- Fully offline audio deepfake detection
- FastAPI backend
- Interactive frontend (drag & drop)
- Privacy-first (no cloud usage)

## How to Run

1. Install Python 3.10+
2. Install dependencies:
   pip install fastapi uvicorn torch librosa numpy python-multipart
3. Start backend:
   python -m uvicorn backend.main:app --reload
4. Open frontend/index.html in browser
5. Upload WAV audio and analyze

## Notes
- Works fully offline
- Uploaded files are processed temporarily

# EdgeGuard AI

Offline Audio Deepfake Detection using Edge AI

## 🚀 Features
- Works fully offline
- Lightweight edge-friendly model
- Detects AI-generated / fake audio
- Web-based UI + FastAPI backend

## 🧠 Architecture
Frontend (HTML/JS) → FastAPI → Offline Audio Engine (PyTorch)

## ⚙️ Setup Instructions

### 1. Clone the repo

git clone git@github.com:umararshad-glitch/edgeguard-ai.git
cd edgeguard_alt

python -m pip install -r requirements.txt

