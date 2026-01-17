import os
import librosa
import numpy as np
import joblib

# CONFIG
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "audio_model.pkl")
REFERENCE_MFCC_PATH = os.path.join(BASE_DIR, "model", "reference_mfcc.npy")

# Load trained model
model = joblib.load(MODEL_PATH)

# Load reference MFCC safely
if os.path.exists(REFERENCE_MFCC_PATH):
    REFERENCE_MFCC = np.load(REFERENCE_MFCC_PATH)
else:
    REFERENCE_MFCC = None


# FEATURE EXTRACTION (FOR PREDICTION)
def extract_features(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc, axis=1)  # shape (20,)


# PREDICTION + EXPLAINABILITY
def analyze_audio(audio_path):
    # --- Extract MFCC for explainability ---
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)  # shape (20,)

    # --- MFCC deviation score ---
    if REFERENCE_MFCC is not None:
        mfcc_diff = np.abs(mfcc_mean - REFERENCE_MFCC)
        mfcc_diff_score = float(np.mean(mfcc_diff))
    else:
        mfcc_diff_score = 0.0

    # --- Prediction (UNCHANGED MODEL LOGIC) ---
    feats = mfcc_mean.reshape(1, -1)
    probs = model.predict_proba(feats)[0]

    fake_prob = probs[1]
    real_prob = probs[0]

    prediction = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = max(fake_prob, real_prob) * 100

    # Confidence calibration
    if confidence > 90 and prediction == "FAKE":
        confidence -= 20

    # --- Return ---
    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "engine": "ASVspoof2019-LA (Trained)",
        "mfcc_deviation_score": round(mfcc_diff_score, 4),
        "explanation": (
            "Abnormal speech modulation detected compared to natural human speech"
            if mfcc_diff_score > 0.15
            else "Speech modulation aligns with natural human patterns"
        )
    }

