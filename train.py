"""
train.py — Full training pipeline with anti-overfitting measures
================================================================
Key improvements over the original:
  1. Proper 80/10/10 train/val/test split (stratified)
  2. SpecAugment-style feature noise augmentation
  3. Label smoothing in loss
  4. CosineAnnealingLR scheduler
  5. Early stopping (patience=5 on val loss)
  6. Saves best model checkpoint (not last)
  7. Logs train vs val metrics to detect overfitting in real-time
"""

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from model import CNNAttentionClassifier


# ── Dataset ───────────────────────────────────────────────────────────────────

class FeatureDataset(Dataset):
    """Wraps pre-extracted (feature, label) pairs with optional augmentation."""

    def __init__(self, features, labels, augment: bool = False):
        self.features = features
        self.labels   = labels
        self.augment  = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx].float()
        y = self.labels[idx]

        if self.augment:
            # Gaussian noise injection (mild)
            x = x + torch.randn_like(x) * 0.02

            # Random feature dropout (like SpecAugment for embeddings)
            mask = torch.rand(x.shape) > 0.05
            x = x * mask.float()

        return x, y


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_features(path: str = "features.pt"):
    raw = torch.load(path)
    features = torch.stack([item[0] for item in raw])
    labels   = torch.tensor([item[1] for item in raw])
    return features, labels


def normalize(features, mean=None, std=None):
    if mean is None:
        mean = features.mean(dim=0)
        std  = features.std(dim=0)
    return (features - mean) / (std + 1e-6), mean, std


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            total_loss += loss.item()
            preds   = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return total_loss / len(loader), correct / total * 100


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    SEED        = 42
    BATCH       = 32
    EPOCHS      = 60
    LR          = 3e-4
    WEIGHT_DECAY= 1e-4
    PATIENCE    = 8      # early stopping patience
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {DEVICE}")

    # Load & normalize
    features, labels = load_features("features.pt")
    print(f"Loaded {len(labels)} samples  |  Class balance: "
          f"Real={( labels==0).sum()}  Fake={(labels==1).sum()}")

    features, mean, std = normalize(features)
    torch.save({"mean": mean, "std": std}, "norm.pt")
    print("Normalization stats saved → norm.pt")

    # Stratified split: 80 / 10 / 10
    idx = np.arange(len(labels))
    idx_train, idx_tmp = train_test_split(
        idx, test_size=0.20, stratify=labels.numpy(), random_state=SEED)
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.50, stratify=labels[idx_tmp].numpy(), random_state=SEED)

    train_ds = FeatureDataset(features[idx_train], labels[idx_train], augment=True)
    val_ds   = FeatureDataset(features[idx_val],   labels[idx_val],   augment=False)
    test_ds  = FeatureDataset(features[idx_test],  labels[idx_test],  augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    print(f"Split  →  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # Model
    model = CNNAttentionClassifier().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # Label smoothing reduces overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Training
    best_val_loss = float("inf")
    patience_ctr  = 0
    best_epoch    = 0

    print("\n" + "─" * 65)
    print(f"{'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}  {'LR':>10}")
    print("─" * 65)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_loss    += loss.item()
            preds       = logits.argmax(dim=1)
            tr_correct += (preds == y).sum().item()
            tr_total   += y.size(0)

        scheduler.step()

        tr_loss /= len(train_loader)
        tr_acc   = tr_correct / tr_total * 100

        va_loss, va_acc = evaluate(model, val_loader, criterion, DEVICE)
        lr_now = scheduler.get_last_lr()[0]

        print(f"{epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>6.2f}%  "
              f"{va_loss:>8.4f}  {va_acc:>6.2f}%  {lr_now:>10.2e}")

        # Checkpoint best model
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch    = epoch
            patience_ctr  = 0
            torch.save(model.state_dict(), "classifier.pth")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n⏹  Early stopping at epoch {epoch}. "
                      f"Best was epoch {best_epoch} (val_loss={best_val_loss:.4f})")
                break

    print("─" * 65)

    # Final test evaluation
    model.load_state_dict(torch.load("classifier.pth"))
    te_loss, te_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\n✅  Test set  →  Loss: {te_loss:.4f}  Accuracy: {te_acc:.2f}%")
    print(f"Best checkpoint: epoch {best_epoch}  (val_loss={best_val_loss:.4f})")
    print("Saved: classifier.pth  norm.pt")


if __name__ == "__main__":
    main()
