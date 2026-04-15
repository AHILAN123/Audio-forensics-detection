import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Load model + processor
print("Loading model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()
print("Model loaded successfully!")

# Load audio
def load_audio(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=16000)  # auto resample
    return torch.tensor(waveform)

# Run model
def run(file_path):
    waveform = load_audio(file_path)

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    print("Output shape:", outputs.last_hidden_state.shape)

if __name__ == "__main__":
    run("test.wav")
