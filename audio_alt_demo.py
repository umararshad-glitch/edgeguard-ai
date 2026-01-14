import torch
import librosa
import numpy as np

AUDIO_PATH = "../test.wav"  # reuse same test file safely

labels = ["REAL", "FAKE"]

# Lightweight offline model (alt version)
model = torch.nn.Sequential(
    torch.nn.Linear(40, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 2),
    torch.nn.Softmax(dim=1)
)

model.eval()

signal, sr = librosa.load(AUDIO_PATH, sr=16000)
mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
mfcc_mean = np.mean(mfcc.T, axis=0)

x = torch.tensor(mfcc_mean).float().unsqueeze(0)

with torch.no_grad():
    output = model(x)
    confidence, prediction = torch.max(output, 1)

print("ALT ENGINE RESULT")
print("Prediction:", labels[prediction.item()])
print("Confidence:", round(confidence.item() * 100, 2), "%")
print("Mode: OFFLINE (Edge)")
