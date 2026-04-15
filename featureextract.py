import os
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

def extract(file_path):
    waveform, _ = librosa.load(file_path, sr=16000)
    waveform = torch.tensor(waveform)

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    features = outputs.last_hidden_state.mean(dim=1).squeeze()
    return features

data = []

for label, folder in enumerate(["real", "fake"]):
    path = os.path.join("DATASET", folder)

    for file in os.listdir(path):
        if file.endswith(".wav"):
            file_path = os.path.join(path, file)
            print("Processing:", file_path)

            feat = extract(file_path)
            data.append((feat, label))

torch.save(data, "features.pt")
print("Features saved!")
