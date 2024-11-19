import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity

def melspectrogram_cosine_similarity(mel1, mel2):
    mel_1 = np.array(mel1)
    mel_2 = np.array(mel2)

    log_mel1 = librosa.power_to_db(mel_1, ref=np.max)
    log_mel2 = librosa.power_to_db(mel_2, ref=np.max)

    vector1 = log_mel1.flatten().reshape(1,-1)
    vector2 = log_mel2.flatten().reshape(1,-1)

    similarity = cosine_similarity(vector1, vector2)

    return similarity


def melspectrogram_mse(mel1, mel2):
    mel_1 = np.array(mel1)
    mel_2 = np.array(mel2)

    log_mel1 = librosa.power_to_db(mel_1, ref=np.max)
    log_mel2 = librosa.power_to_db(mel_2, ref=np.max)

    min_shape = (min(log_mel1.shape[0], log_mel2.shape[0]), min(log_mel1.shape[1], log_mel2.shape[1]))
    log_mel1 = log_mel1[:min_shape[0], :min_shape[1]]
    log_mel2 = log_mel2[:min_shape[0], :min_shape[1]]

    mse = np.mean((log_mel1 - log_mel2) ** 2)

    return mse