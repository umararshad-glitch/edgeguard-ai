import os
import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# CONFIG 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "ASVspoof2019_LA")
TRAIN_DIR = os.path.join(DATA_DIR, "ASVspoof2019_LA_train", "flac")
PROTOCOL = os.path.join(
    DATA_DIR,
    "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019.LA.cm.train.trn.txt"
)

MODEL_OUT = os.path.join(os.path.dirname(__file__), "audio_model.pkl")

# FEATURE EXTRACTION 
def extract_features(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc.T, axis=0)

#  LOAD DATA 
X, y = [], []

print("Loading ASVspoof protocol...")
with open(PROTOCOL, "r") as f:
    for line in f:
        parts = line.strip().split()
        file_id = parts[1]
        label = parts[-1]

        audio_path = os.path.join(TRAIN_DIR, file_id + ".flac")
        if not os.path.exists(audio_path):
            continue

        feats = extract_features(audio_path)
        X.append(feats)
        y.append(0 if label == "bonafide" else 1)

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples")

#  TRAIN MODEL 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

print("Training model...")
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Validation accuracy: {acc:.4f}")

#  SAVE MODEL 
joblib.dump(model, MODEL_OUT)
print(f"Model saved as {MODEL_OUT}")
