import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram


def pad_to_fixed_length(waveform, fixed_length):
    current_length = waveform.shape[1]
    if current_length < fixed_length:
        waveform = F.pad(waveform, (0, fixed_length - current_length))

    return waveform


def to_melspectrogram(waveform, sr):
    return torchaudio.transforms.AmplitudeToDB()(
        MelSpectrogram(
            sample_rate=sr,
            n_mels=30,
        )(waveform)
    )[0]


def collate_function(batch, fixed_length):
    sample_rates = [item[1] for item in batch]
    padded_waveforms = torch.stack(
        [
            to_melspectrogram(pad_to_fixed_length(item[0], fixed_length), item[1])
            for item in batch
        ]
    )

    return padded_waveforms, torch.tensor(sample_rates)
