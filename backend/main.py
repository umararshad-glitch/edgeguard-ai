import sys
import os
import shutil

# ---------------- PATH FIX (CRITICAL) ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# ---------------- FASTAPI ----------------
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ---------------- MODEL IMPORT ----------------
from model.audio_engine import analyze_audio
from model.video_engine import analyze_video


# ---------------- APP SETUP ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- UPLOADS ----------------
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- ROUTES ----------------
@app.post("/analyze-audio")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = analyze_audio(file_path)
    return result
@app.post("/analyze-video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = analyze_video(file_path)
    return result


