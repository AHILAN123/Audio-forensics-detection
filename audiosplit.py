import librosa
import soundfile as sf
import os

def split_audio(input_folder, output_folder, label, duration=7):
    os.makedirs(output_folder, exist_ok=True)
    count = 0
    
    if not os.path.exists(input_folder):
        print(f" Folder not found: {input_folder}")
        return

    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            path = os.path.join(input_folder, file)

            print(f"Processing: {file}")

            try:
                audio, sr = librosa.load(path, sr=16000)
                chunk_size = duration * sr 
                
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i+chunk_size]
                    if len(chunk) < chunk_size:
                        continue

                    filename = f"{label}_{count}.wav"
                    output_path = os.path.join(output_folder, filename)

                    sf.write(output_path, chunk, sr)

                    count += 1

            except Exception as e:
                print(f"Error processing {file}: {e}")

    print(f" Done processing {label} files. Total chunks: {count}")




if __name__ == "__main__":
    split_audio("DATASET/Fake", "dataset/fake", "fake", duration=7)
    split_audio("DATASET/Real", "dataset/real", "real", duration=7)

    print("All splitting completed successfully!")
