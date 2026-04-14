import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#data comes here : - 
def load_data(folder, label, vocoder_label):
    data = []
    labels = []
    vocoders = []

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            path = os.path.join(folder, file)

            TARGET_WIDTH = 300

            spec = np.load(path)

            # Normalize
            spec = (spec - np.mean(spec)) / np.std(spec)

            # Fix width
            if spec.shape[1] < TARGET_WIDTH:
                pad_width = TARGET_WIDTH - spec.shape[1]
                spec = np.pad(spec, ((0, 0), (0, pad_width)))
            else:
                spec = spec[:, :TARGET_WIDTH]

            data.append(spec)
            labels.append(label)
            vocoders.append(vocoder_label)

    return data, labels, vocoders

real_data, real_labels, real_voc = load_data("spectrograms/real", 0, 0)
fake_data, fake_labels, fake_voc = load_data("spectrograms/fake", 1, 1)

X = np.array(real_data + fake_data)
y = np.array(real_labels + fake_labels)
v = np.array(real_voc + fake_voc)

# Add channel dimension
X = np.expand_dims(X, axis=1)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
v = torch.tensor(v, dtype=torch.long)



#our model: - 
class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        weights = torch.sigmoid(self.conv(x))
        return x * weights


class AdvancedCNN(nn.Module):
    def __init__(self, num_vocoders=2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.attention = Attention(128)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        # now size is fixed → no guesswork
        self.fc_shared = nn.Linear(128 * 8 * 8, 256)

        self.fc_real_fake = nn.Linear(256, 1)
        self.fc_vocoder = nn.Linear(256, num_vocoders)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.attention(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc_shared(x))

        real_fake = torch.sigmoid(self.fc_real_fake(x))
        vocoder = self.fc_vocoder(x)

        return real_fake, vocoder

model = AdvancedCNN()


# -----------------------------
# LOSS + OPTIMIZER
# -----------------------------

criterion_binary = nn.BCELoss()
criterion_multi = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


# -----------------------------
# TRAINING LOOP
# -----------------------------

epochs = 5

for epoch in range(epochs):

    real_fake_pred, vocoder_pred = model(X)

    real_fake_pred = real_fake_pred.squeeze()

    loss1 = criterion_binary(real_fake_pred, y)
    loss2 = criterion_multi(vocoder_pred, v)

    loss = loss1 + loss2  # multi-task loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")



torch.save(model.state_dict(), "advanced_model.pth")

print("Training done")