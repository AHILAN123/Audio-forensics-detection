from flask import Flask, request, jsonify
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch import nn
import tempfile

app = Flask(__name__)

# Load models
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec.eval()

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

model = Classifier()
model.load_state_dict(torch.load("classifier.pth"))
model.eval()

norm = torch.load("norm.pt")
mean = norm["mean"]
std = norm["std"]

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        waveform, _ = librosa.load(tmp.name, sr=16000)

    waveform = torch.tensor(waveform)

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = wav2vec(**inputs)

    features = outputs.last_hidden_state.mean(dim=1)
    features = (features - mean) / (std + 1e-6)

    logits = model(features)
    probs = torch.softmax(logits, dim=1)

    real_prob = probs[0][0].item()
    fake_prob = probs[0][1].item()

    result = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = max(real_prob, fake_prob)

    return jsonify({
        "result": result,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
