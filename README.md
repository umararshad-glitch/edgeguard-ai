# EdgeGuard AI

**Offline Audio & Video Deepfake Detection System**

EdgeGuard AI is a privacy-first, fully offline deepfake detection system designed for edge devices.  
It supports both **audio** and **video** deepfake analysis through a unified interface.

---

## 🚀 Features

- ✅ Fully offline processing (no cloud, no APIs)
- 🎧 Audio deepfake detection (WAV files)
- 🎥 Video deepfake detection (MP4 files)
- ⚖️ Outputs REAL / FAKE / UNCERTAIN with confidence
- 🌐 FastAPI backend
- 🎨 Lightweight, modern web frontend
- 🔒 Privacy-first (files processed locally)

---

## 🧠 System Architecture

      Frontend (HTML / JS)
              ↓
      FastAPI Backend
              ↓
      Audio Engine (PyTorch)
      Video Engine (OpenCV + MediaPipe)


Each engine works independently and is optimized for offline, edge-friendly execution.

---

## ⚙️ Requirements

- Python **3.10 or 3.11** (recommended)
- Windows / Linux / macOS
- No GPU required

---

## 📦 Installation

### 1️⃣ Clone the repository

1. git clone https://github.com/umararshad-glitch/edgeguard-ai.git
2. cd edgeguard_alt

2️⃣ Install dependencies

py -3.11 -m pip install -r requirements.txt

If installing manually:

py -3.11 -m pip install fastapi uvicorn torch librosa numpy python-multipart opencv-python mediapipe

▶️ How to Run

1️⃣ Start the backend

py -3.11 -m uvicorn backend.main:app --reload

2️⃣ Open the frontend

Open this file in your browser:

frontend/index.html

🧪 How to Use

1. Select Audio or Video
2. Upload a file:
   Audio: .wav
   Video: .mp4
3. Click Analyze
4. View prediction and confidence

📊 Output Labels

REAL → High confidence genuine media
FAKE → High confidence synthetic / manipulated media
UNCERTAIN → Low confidence (ambiguous input)

The system is intentionally conservative to reduce false positives.

🎥 Video Detection Notes

1. Video detection is a lightweight offline module
2. Uses facial consistency analysis across frames
3. Designed for edge devices and demo scenarios
4. Not a cloud-based or heavy deep learning pipeline

🔒 Privacy & Security

1. Files are processed locally
2. No data is stored or uploaded
3. No internet required after setup
