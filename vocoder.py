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

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def mel_to_wave(mels):
    # 確保輸入是張量
    if isinstance(mels, list):
        mels = [torch.tensor(mel, dtype=torch.float32) if isinstance(mel, np.ndarray) else mel for mel in mels]
        mels = torch.stack(mels)
    elif isinstance(mels, np.ndarray):
        mels = torch.tensor(mels, dtype=torch.float32)

    mels = mels.to(torch.float32)
    
    # 確保是 4D Tensor (batch_size, num_channels, freq_bins, timesteps)
    if mels.ndimension() != 4:
        raise ValueError("Input mel spectrogram must be a 4D tensor (batch_size, num_channels, freq_bins, timesteps)")

    # 移除 num_channels 維度 (從 (B, 1, F, T) -> (B, F, T))
    if mels.size(1) == 1:
        mels = mels.squeeze(1)

    # 將頻率維度調整為 80（HiFi-GAN 的要求）
    if mels.size(1) != 80:
        # 暫時增加一個 channel 維度以滿足 interpolate 的要求
        mels = mels.unsqueeze(1)  # (B, 1, F, T)
        mels = F.interpolate(mels, size=(80, mels.size(-1)), mode="bilinear", align_corners=False)
        mels = mels.squeeze(1)  # (B, F, T)

    # HiFi-GAN 解碼需要的格式是 (batch_size, freq_bins=80, timesteps)
    waveforms = hifi_gan.decode_batch(mels)  

    return waveforms

# 測試輸入
a = torch.randn(2, 1, 32, 4096)  # 原始輸入

# 呼叫函數
waveforms = mel_to_wave(a)

# 確保 waveforms 是 PyTorch tensor
if isinstance(waveforms, list):  # HiFi-GAN 輸出可能是列表格式
    waveforms = torch.cat(waveforms, dim=0)

# 提取 batch 中的第一個 waveform
waveform = waveforms[0]  # 第一個 waveform，形狀 (channels=1, time_steps)

# 如果還有多餘的 batch 維度，進一步移除
if waveform.ndimension() == 3:  # (batch_size, channels, time_steps)
    waveform = waveform.squeeze(0)  # 移除 batch_size 維度，變成 (channels, time_steps)

# 保存為 .wav 文件
filename = "output.wav"
sample_rate = 22050  # HiFi-GAN 的輸出通常是 22050 Hz
torchaudio.save(filename, waveform, sample_rate)  # 傳遞正確形狀的 waveform
print(f"Saved waveform as {filename}")