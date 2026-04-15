import torch
from torch import nn
from torch.utils.data import DataLoader

# Load features
data = torch.load("features.pt")

# Split features and labels
features = torch.stack([item[0] for item in data])
labels = torch.tensor([item[1] for item in data])

# 🔥 Compute normalization stats
mean = features.mean(dim=0)
std = features.std(dim=0)

# Save stats
torch.save({"mean": mean, "std": std}, "norm.pt")

# Normalize features
features = (features - mean) / (std + 1e-6)

# Dataset
dataset = list(zip(features, labels))
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
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

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

# Training loop
for epoch in range(20):
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = (correct / total) * 100

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "classifier.pth")
print("Classifier trained and saved!")
