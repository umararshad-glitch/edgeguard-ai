# EdgeGuard AI

**Offline Audio Deepfake Detection using Edge AI**

> Core audio analysis and inference engine implemented in **Python (FastAPI + PyTorch)**.  
> Frontend is a **lightweight web interface** for easy interaction.

---

## 🔍 What is EdgeGuard AI?

**EdgeGuard AI** is an **offline-first audio deepfake detection system** that identifies whether an audio sample is **REAL or AI-generated (FAKE)**.

The system runs entirely on the **local device (edge)** without sending any data to the cloud, ensuring **privacy, low latency, and usability in low-connectivity environments**.

---

## ✨ Key Features

- ✅ Fully **offline** audio deepfake detection  
- 🧠 Python-based inference engine (**PyTorch**)  
- ⚡ FastAPI backend for real-time analysis  
- 🖥️ Simple web UI (drag & drop audio upload)  
- 🔒 Privacy-first (no cloud usage, no permanent storage)

---

## 🧠 How It Works

1. User uploads a **WAV audio file** via the web interface  
2. Frontend sends the audio to the **FastAPI backend**  
3. Backend extracts audio features and runs an **offline PyTorch model**  
4. System returns:
   - Prediction: **REAL / FAKE**
   - Confidence score  
5. Result is displayed instantly in the browser  

**Pipeline:**

Frontend (HTML/JS)
↓
FastAPI Backend
↓
Offline Audio Engine (PyTorch)


---

## 🚀 How to Run EdgeGuard AI 

### 🔹 Prerequisites
- Python **3.10 or higher**
- Git
- Any modern web browser

---

### 🔹 Step 1: Clone the Repository

git clone https://github.com/umararshad-glitch/edgeguard-ai.git
cd edgeguard_alt

🔹 Step 2: Install Dependencies (One Time)
python -m pip install -r requirements.txt

🔹 Step 3: Start the Backend Server
python -m uvicorn backend.main:app --reload

🔹 Step 4: Open the Frontend

Open this file in your browser:
frontend/index.html

🔹 Step 5: Test the System

1. Upload any WAV audio file
2. Click Analyze Audio
3. View prediction and confidence


🖱️ One-Click Run (Windows)

For easier access on Windows:
Open the project folder in File Explorer
Double-click:
run.bat
Backend starts automatically
Open frontend/index.html in browser

🔒 Privacy & Offline Mode
All audio is processed locally
No files are stored permanently
No internet connection required after setup

🎯 Use Cases
Detecting AI-generated or manipulated audio
Testing synthetic voices
Educational demos on audio deepfakes
Privacy-focused edge AI applications

🧪 Notes
Recommended audio format: WAV
Lightweight model designed for edge usage
Suitable for demos, research, and hackathons
