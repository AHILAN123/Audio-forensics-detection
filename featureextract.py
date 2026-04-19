"""
featureextract.py — Extract Wav2Vec2 features from dataset
===========================================================
Processes real/ and fake/ subfolders and saves features.pt
"""

import os
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

DATASET_ROOT = "DATASET"   # change if your folder is named differently
OUTPUT_FILE  = "features.pt"
SR           = 16000
CHUNK_SEC    = 7           # max seconds per clip (matches audiosplit.py)

print("Loading Wav2Vec2-base...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(DEVICE)
print(f"Using device: {DEVICE}")


def extract_features(file_path: str) -> torch.Tensor:
    waveform, _ = librosa.load(file_path, sr=SR, duration=CHUNK_SEC)
    waveform    = torch.tensor(waveform)

    inputs = processor(waveform, sampling_rate=SR, return_tensors="pt",
                       padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean-pool over time → 768-dim vector
    feat = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
    return feat


data = []
label_map = {"real": 0, "fake": 1}   # 0=real, 1=fake

for label_name, label_idx in label_map.items():
    folder = os.path.join(DATASET_ROOT, label_name)

    if not os.path.exists(folder):
        print(f"⚠  Folder not found: {folder}  (skipping)")
        continue

    files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
    print(f"\nProcessing {label_name}: {len(files)} files")

    for i, fname in enumerate(files, 1):
        fpath = os.path.join(folder, fname)
        try:
            feat = extract_features(fpath)
            data.append((feat, label_idx))
            if i % 20 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {fname}")
        except Exception as e:
            print(f"  ✗ Error: {fname} → {e}")

torch.save(data, OUTPUT_FILE)
print(f"\n✅  Saved {len(data)} feature vectors → {OUTPUT_FILE}")
