import sounddevice as sd
from scipy.io.wavfile import write
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch import nn

# Record audio
def record_audio(filename="input.wav", duration=5, fs=16000):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("✅ Recording finished!")

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

# Load classifier
model = Classifier()
model.load_state_dict(torch.load("classifier.pth"))
model.eval()

# Load normalization
norm = torch.load("norm.pt")
mean = norm["mean"]
std = norm["std"]

# Prediction
def predict(file_path):
    waveform, _ = librosa.load(file_path, sr=16000)
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

    print("\n🔍 RESULT:")
    if fake_prob > real_prob:
        print(f"⚠ FAKE ({fake_prob*100:.2f}%)")
    else:
        print(f"✅ REAL ({real_prob*100:.2f}%)")

# Run
record_audio()
predict("input.wav")
