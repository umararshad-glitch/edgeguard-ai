import os
import librosa
import numpy as np
import joblib



# =====================
# CONFIG
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "audio_model.pkl")
REFERENCE_MFCC_PATH = os.path.join(BASE_DIR, "model", "reference_mfcc.npy")
REFERENCE_MFCC_PATH = os.path.join(BASE_DIR, "model", "reference_mfcc.npy")

if os.path.exists(REFERENCE_MFCC_PATH):
    REFERENCE_MFCC = np.load(REFERENCE_MFCC_PATH)
else:
    REFERENCE_MFCC = None


# Load trained model
model = joblib.load(MODEL_PATH)

# Load reference MFCC safely (for explainability only)
if os.path.exists(REFERENCE_MFCC_PATH):
    REFERENCE_MFCC = np.load(REFERENCE_MFCC_PATH)
else:
    REFERENCE_MFCC = None


# =====================
# FEATURE EXTRACTION
# =====================
def extract_features(path):
    """
    Extract MFCC features for prediction
    """
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc, axis=1)  # shape (20,)


# =====================
# ANALYSIS
# =====================
def analyze_audio(audio_path):
    # -------- Explainability (MFCC deviation) --------
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)

    if REFERENCE_MFCC is not None:
        mfcc_diff = np.abs(mfcc_mean - REFERENCE_MFCC)
        mfcc_diff_score = float(np.mean(mfcc_diff))
    else:
        mfcc_diff_score = 0.0

    # -------- Prediction (MODEL OUTPUT) --------
    feats = mfcc_mean.reshape(1, -1)
    probs = model.predict_proba(feats)[0]

    real_prob = probs[0]
    fake_prob = probs[1]

    # -------- CONSERVATIVE DECISION LOGIC --------
    # Avoid false positives on real user audio
# -------- 3-CLASS DECISION LOGIC --------
# Conservative, forensic-style thresholds



# -------- 3-CLASS DECISION LOGIC --------
    if fake_prob > 0.75:
     prediction = "FAKE"
     confidence = fake_prob * 100

    elif real_prob > 0.75:
     prediction = "REAL"
     confidence = real_prob * 100

    else:
     prediction = "UNCERTAIN"
     confidence = max(real_prob, fake_prob) * 100

# -------- Confidence calibration --------
    if prediction == "FAKE" and confidence > 90:
       confidence -= 20

    # -------- FINAL RESPONSE --------\
    mfcc_diff_map = mfcc_diff.tolist() if REFERENCE_MFCC is not None else []

    return {
    "prediction": prediction,
    "confidence": round(confidence, 2),
    "engine": "ASVspoof2019-LA (Trained)",
    "mfcc_deviation_score": round(mfcc_diff_score, 4),
    "mfcc_diff_map": mfcc_diff_map,
    "explanation": (
        "Strong abnormal speech modulation detected compared to natural human speech"
        if prediction == "FAKE"
        else "Speech modulation aligns with natural human patterns"
        if prediction == "REAL"
        else "Audio characteristics fall in an ambiguous zone; further verification recommended"
    )
}


    
