import torch
import librosa
import numpy as np

labels = ["REAL", "FAKE"]

# Simple offline demo model
model = torch.nn.Sequential(
    torch.nn.Linear(40, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 2),
    torch.nn.Softmax(dim=1)
)

model.eval()

def analyze_audio(audio_path):
    # Load audio
    signal, sr = librosa.load(audio_path, sr=16000)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Convert to tensor
    x = torch.tensor(mfcc_mean).float().unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        confidence, prediction = torch.max(output, dim=1)

    confidence_value = confidence.item()

    # 🔹 Confidence-based decision
    if confidence_value < 0.60:
        final_label = "UNCERTAIN"
    else:
        final_label = labels[prediction.item()]

    return {
        "engine": "EdgeGuard-ALT",
        "mode": "OFFLINE",
        "prediction": final_label,
        "confidence": round(confidence_value * 100, 2)
    }
