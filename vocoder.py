import numpy as np
import torch
import torch.nn.functional as F
from speechbrain.inference.vocoders import HIFIGAN


hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir="pretrained_models/tts-hifigan-ljspeech",
)


def mel_to_wave(mels):
    if isinstance(mels, list):
        mels = [
            (
                torch.tensor(mel, dtype=torch.float32)
                if isinstance(mel, np.ndarray)
                else mel
            )
            for mel in mels
        ]
        mels = torch.stack(mels)
    elif isinstance(mels, np.ndarray):
        mels = torch.tensor(mels, dtype=torch.float32)

    mels = mels.to(torch.float32)

    if mels.ndimension() != 4:
        raise ValueError(
            "Input mel spectrogram must be a 4D tensor (batch_size, num_channels, freq_bins, timesteps)"
        )

    if mels.size(1) == 1:
        mels = mels.squeeze(1)

    if mels.size(1) != 80:
        mels = mels.unsqueeze(1)
        mels = F.interpolate(
            mels, size=(80, mels.size(-1)), mode="bilinear", align_corners=False
        )
        mels = mels.squeeze(1)

    waveforms = hifi_gan.decode_batch(mels)

    return waveforms


a = torch.randn(2, 1, 32, 4096)
