"""
app.py  —  VoiceGuard FastAPI Backend
======================================
Install deps:
    pip install fastapi uvicorn python-multipart librosa torch transformers soundfile

Run:
    python app.py

Server starts at  http://localhost:8000
Open index.html separately (double-click or serve with: python -m http.server 5500)
"""

import os, time, tempfile
import torch, librosa
from torch import nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(BASE_DIR, "classifier.pth")
NORM_PATH       = os.path.join(BASE_DIR, "norm.pt")
SR              = 16000
CHUNK_SEC       = 7


# ── model (identical to model.py / train.py) ──────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = self.dropout((q @ k.transpose(-2,-1) * self.scale).softmax(dim=-1))
        return self.out_proj((attn @ v).transpose(1,2).reshape(B, T, C))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)) + self.res(x))


class CNNAttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(1, 32, 7), nn.MaxPool1d(2),
            ConvBlock(32, 64, 5), nn.MaxPool1d(2),
            ConvBlock(64, 128, 3), nn.MaxPool1d(2),
        )
        self.norm1     = nn.LayerNorm(128)
        self.attn      = MultiHeadSelfAttention(128, 4)
        self.norm2     = nn.LayerNorm(128)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        self.head = nn.Sequential(
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 64),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x).permute(0, 2, 1)
        x = x + self.attn(self.norm1(x))
        x = self.norm2(x).mean(dim=1)
        return self.head(x)


# ── load at startup ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[VoiceGuard] Device: {DEVICE}")
print("[VoiceGuard] Loading Wav2Vec2-base ...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec   = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
wav2vec.eval()
print("[VoiceGuard] Loading classifier ...")
clf = CNNAttentionClassifier().to(DEVICE)
clf.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
clf.eval()
norm = torch.load(NORM_PATH, map_location=DEVICE)
MEAN = norm["mean"].to(DEVICE)
STD  = norm["std"].to(DEVICE)
print("[VoiceGuard] Ready — http://localhost:8000\n")


# ── app ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="VoiceGuard", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET","POST"], allow_headers=["*"])


@app.get("/")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    t0 = time.time()
    suffix = os.path.splitext(file.filename or "audio.wav")[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        waveform, _ = librosa.load(tmp_path, sr=SR, duration=CHUNK_SEC, mono=True)
        duration_s  = round(len(waveform) / SR, 2)
        if len(waveform) < 1600:
            raise HTTPException(422, "Audio too short")
        wt     = torch.tensor(waveform)
        inputs = processor(wt, sampling_rate=SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            feats  = wav2vec(**inputs).last_hidden_state.mean(dim=1)
            feats  = (feats - MEAN) / (STD + 1e-6)
            logits = clf(feats)
            probs  = torch.softmax(logits, dim=1)[0]
        real_prob  = round(probs[0].item() * 100, 2)
        fake_prob  = round(probs[1].item() * 100, 2)
        confidence = round(max(real_prob, fake_prob), 2)
        label      = "FAKE" if fake_prob > real_prob else "REAL"
        risk = ("HIGH" if confidence >= 80 else "MEDIUM") if label == "FAKE" else ("LOW" if confidence >= 70 else "MEDIUM")
        return JSONResponse({
            "label": label, "real_prob": real_prob, "fake_prob": fake_prob,
            "confidence": confidence, "risk_level": risk,
            "duration_s": duration_s, "process_ms": int((time.time()-t0)*1000),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
