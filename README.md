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
