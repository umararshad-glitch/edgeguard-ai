import os
import librosa
import numpy as np
import joblib

#  CONFIG 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "audio_model.pkl")

# Load trained model
model = joblib.load(MODEL_PATH)

#  FEATURE EXTRACTION 
def extract_features(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc.T, axis=0)

# PREDICTION 
def analyze_audio(audio_path):
    feats = extract_features(audio_path).reshape(1, -1)
    probs = model.predict_proba(feats)[0]

    fake_prob = probs[1]
    real_prob = probs[0]

    prediction = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = max(fake_prob, real_prob) * 100

    #  Confidence calibration 
    if confidence > 90 and prediction == "FAKE":
        confidence -= 20  # dampen overconfidence for real-world audio

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "engine": "ASVspoof2019-LA (Trained)"
    }

