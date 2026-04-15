import os
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Load pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec.eval()

# Freeze wav2vec (important)
for param in wav2vec.parameters():
    param.requires_grad = False

# Dataset class
class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        
        for label, folder in enumerate(["real", "fake"]):
            path = os.path.join(root_dir, folder)
            for file in os.listdir(path):
                if file.endswith(".wav"):
                    self.files.append((os.path.join(path, file), label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]

        waveform, _ = librosa.load(file_path, sr=16000)
        waveform = torch.tensor(waveform)

        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = wav2vec(**inputs)

        features = outputs.last_hidden_state.mean(dim=1).squeeze()

        return features, torch.tensor(label)

# Classifier
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        return self.fc(x)

# Load dataset
dataset = AudioDataset("DATASET")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(5):
    total_loss = 0

    for features, labels in loader:
        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "classifier.pth")
print("Model saved!")
