import torch
import torch.nn.functional as F


def pad_to_fixed_length(waveform, fixed_length):
    current_length = waveform.shape[1]
    if current_length < fixed_length:
        waveform = F.pad(waveform, (0, fixed_length - current_length))

    return waveform


def collate_fn_fixed(batch, fixed_length):
    waveforms = [item[0] for item in batch]
    sample_rates = [item[1] for item in batch]
    padded_waveforms = torch.stack(
        [pad_to_fixed_length(w, fixed_length) for w in waveforms]
    )

    return padded_waveforms, torch.tensor(sample_rates)
