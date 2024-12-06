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


def mel_to_wave(mels):

    # convert input to tensor
    if isinstance(mels, list):
        mels = [torch.tensor(mel, dtype=torch.float32) if isinstance(mel, np.ndarray) else mel for mel in mels]
        mels = torch.stack(mels)
    elif isinstance(mels, np.ndarray):
        mels = torch.tensor(mels, dtype=torch.float32)

    mels = mels.to(torch.float32)
    
    # this means input is a single mel
    if mels.ndimension() == 2:
        mels = mels.unsqueeze(0)

    # HIFIGAN requires mel's chanel to be 80
    if mels.size(1) != 80:  
        mels = mels[:80, :]

    
    waveforms = hifi_gan.decode_batch(mels)  

    waveforms = [wf.unsqueeze(0) if wf.ndimension() == 1 else wf for wf in waveforms]
    return waveforms 
