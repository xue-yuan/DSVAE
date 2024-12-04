import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F


def melspectrogram_cosine_similarity(mel1, mel2):

    mel1 = torch.tensor(mel1, dtype=torch.float32)
    mel2 = torch.tensor(mel2, dtype=torch.float32)

    log_mel1 = torch.log10(mel1 + 1e-8)
    log_mel2 = torch.log10(mel2 + 1e-8)

    vector1 = log_mel1.flatten()
    vector2 = log_mel2.flatten()

    similarity = F.cosine_similarity(vector1.unsqueeze(0), vector2.unsqueeze(0))

    return similarity


def melspectrogram_mse(mel1, mel2):

    mel1 = torch.tensor(mel1, dtype=torch.float32)
    mel2 = torch.tensor(mel2, dtype=torch.float32)
    
    log_mel1 = torch.log10(mel1 + 1e-8)
    log_mel2 = torch.log10(mel2 + 1e-8)

    min_shape = (
        min(log_mel1.shape[0], log_mel2.shape[0]),
        min(log_mel1.shape[1], log_mel2.shape[1]),
    )
    log_mel1 = log_mel1[: min_shape[0], : min_shape[1]]
    log_mel2 = log_mel2[: min_shape[0], : min_shape[1]]

    mse = F.mse_loss(log_mel1, log_mel2)

    return mse
