import librosa
import numpy as np
import os

def generate_spectrograms(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    count = 0

    for file in os.listdir(input_folder):
        if file.lower().endswith(".wav"):
            path = os.path.join(input_folder, file)

            try:
                # Load audio
                audio, sr = librosa.load(path, sr=16000)

                # Generate mel spectrogram
                spec = librosa.feature.melspectrogram(y=audio, sr=sr)

                # Convert to log scale
                spec_db = librosa.power_to_db(spec, ref=np.max)

                # Save as numpy file
                save_path = os.path.join(output_folder, file.replace(".wav", ".npy"))
                np.save(save_path, spec_db)

                count += 1

            except Exception as e:
                print(f"Error: {file} → {e}")

    print(f"{count} spectrograms created in {output_folder}")

if __name__ == "__main__":
    generate_spectrograms("dataset/real", "spectrograms/real")
    generate_spectrograms("dataset/fake", "spectrograms/fake")

    print("Spectrogram generation complete!")
