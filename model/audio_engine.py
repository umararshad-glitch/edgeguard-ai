import torch
import librosa
import numpy as np

labels = ["REAL", "FAKE"]

model = torch.nn.Sequential(
    torch.nn.Linear(40, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 2),
    torch.nn.Softmax(dim=1)
)

model.eval()

def analyze_audio(audio_path):
    signal, sr = librosa.load(audio_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    x = torch.tensor(mfcc_mean).float().unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        confidence, prediction = torch.max(output, 1)

    return {
        "engine": "EdgeGuard-ALT",
        "mode": "OFFLINE",
        "prediction": labels[prediction.item()],
        "confidence": round(confidence.item() * 100, 2)
    }
