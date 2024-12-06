import torch
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import torchaudio
from speechbrain.inference.TTS import Tacotron2
import numpy as np

#initialize TTS and vocoder
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_models/tts-hifigan-ljspeech")


# running TTS
mel_output, mel_length, alignment = tacotron2.encode_text("Merry christmas cock suckazzzz")

# run vocoder
waveform_2 = hifi_gan.decode_batch(mel_output)
torchaudio.save('ganlinlaoshi.wav', waveform_2.squeeze(1), 22050)


def mel_to_wave(mel):
    
    if isinstance(mel, np.ndarray):
        mel = torch.tensor(mel)

    mel = mel.to(torch.float32)

    if mel.size(0) != 80:  # 如果頻道數不是 80，進行調整
        # 切片，只取前 80 個 mel 頻帶
        mel = mel[:80, :]

    mel = mel.unsqueeze(0) if mel.ndimension() == 2 else mel
    
    wave_form = hifi_gan.decode_batch(mel)

    return wave_form 
#load audio file
# signal, rate = torchaudio.load('speechbrain/tts-hifigan-ljspeech/example.wav')


# Compute mel spectrogram, have to use these specific parameters
# spectrogram, _ = mel_spectogram(
#     audio=signal.squeeze(),
#     sample_rate=22050,
#     hop_length=256,
#     win_length=None,
#     n_mels=80,
#     n_fft=1024,
#     f_min=0.0,
#     f_max=8000.0,
#     power=1,
#     normalized=False,
#     min_max_energy_norm=True,
#     norm="slaney",
#     mel_scale="slaney",
#     compression=True
# )
# generate random mel_spec

# mel_spec = torch.rand(2, 80, 298)

# convert the melspetroram to wavefrom
# waveforms = hifi_gan.decode_batch(mel_spec)

# torchaudio.save('waveform-reconstructed.wav', waveforms.squeeze(1), 22050)

