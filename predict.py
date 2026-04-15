import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch import nn

# Load wav2vec
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec.eval()

# Classifier (same as training)
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

# Load model
model = Classifier()
model.load_state_dict(torch.load("classifier.pth"))
model.eval()

# 🔥 Load normalization stats
norm = torch.load("norm.pt")
mean = norm["mean"]
std = norm["std"]

# Prediction function
def predict(file_path):
    # Load audio
    waveform, _ = librosa.load(file_path, sr=16000)
    waveform = torch.tensor(waveform)

    # Extract features
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = wav2vec(**inputs)

    features = outputs.last_hidden_state.mean(dim=1)

    # 🔥 Apply SAME normalization as training
    features = (features - mean) / (std + 1e-6)

    # Predict
    logits = model(features)
    probs = torch.softmax(logits, dim=1)

    real_prob = probs[0][0].item()
    fake_prob = probs[0][1].item()

    print("\nRESULT:")
    if fake_prob > real_prob:
        print(f"⚠ FAKE ({fake_prob*100:.2f}%)")
    else:
        print(f"✅ REAL ({real_prob*100:.2f}%)")

# Run test
if __name__ == "__main__":
    predict("test.wav")
