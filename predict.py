"""
predict.py — Predict Real / Fake for any .wav file
====================================================
Usage:
  python predict.py path/to/audio.wav
"""

import sys
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from model import CNNAttentionClassifier

SR        = 16000
CHUNK_SEC = 7

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading Wav2Vec2-base...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec   = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec.eval()

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec = wav2vec.to(DEVICE)

print("Loading classifier...")
clf = CNNAttentionClassifier()
clf.load_state_dict(torch.load("classifier.pth", map_location=DEVICE))
clf.eval()
clf = clf.to(DEVICE)

norm = torch.load("norm.pt", map_location=DEVICE)
mean = norm["mean"]
std  = norm["std"]


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(file_path: str) -> dict:
    waveform, _ = librosa.load(file_path, sr=SR, duration=CHUNK_SEC)
    waveform    = torch.tensor(waveform)

    inputs = processor(waveform, sampling_rate=SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs  = wav2vec(**inputs)
        features = outputs.last_hidden_state.mean(dim=1)
        features = (features - mean) / (std + 1e-6)
        logits   = clf(features)
        probs    = torch.softmax(logits, dim=1)[0]

    real_prob = probs[0].item()
    fake_prob = probs[1].item()
    label     = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = max(real_prob, fake_prob) * 100

    return {
        "label":      label,
        "confidence": confidence,
        "real_prob":  real_prob * 100,
        "fake_prob":  fake_prob * 100,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
    result = predict(path)

    print(f"\n{'─'*40}")
    print(f"  File      : {path}")
    print(f"  Result    : {result['label']}")
    print(f"  Confidence: {result['confidence']:.1f}%")
    print(f"  Real prob : {result['real_prob']:.1f}%")
    print(f"  Fake prob : {result['fake_prob']:.1f}%")
    print(f"{'─'*40}")

    if result['label'] == "FAKE":
        print("  ⚠  AI-generated voice detected!")
    else:
        print("  ✅  Human voice — no spoofing detected.")
