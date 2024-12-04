import numpy as np
import librosa
import matplotlib.pyplot as plt
from teacherstudent.train import melspectrogram_cosine_similarity, melspectrogram_mse 



def generate_random_wav(duration=1.0, sr=16000):
    num_samples = int(duration * sr)
    return np.random.uniform(low=-1.0, high=1.0, size=num_samples), sr

def show_melspectrogram(mel, sr, title="Mel Spectrogram"):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), 
                             sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    y1, sr1 = generate_random_wav()
    y2, sr2 = generate_random_wav()

    mel_1 = librosa.feature.melspectrogram(y=y1, sr=sr1, n_mels=128, fmax=8000)
    mel_2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128, fmax=8000)

    show_melspectrogram(mel_1, sr1, title="Mel Spectrogram 1")
    show_melspectrogram(mel_2, sr2, title="Mel Spectrogram 2")

    similarity = melspectrogram_cosine_similarity(mel_1, mel_2)
    print(f"Cosine similarity: {similarity.item()}")

    mse = melspectrogram_mse(mel_1, mel_2)
    print(f"mean squared error: {mse.item()}")
