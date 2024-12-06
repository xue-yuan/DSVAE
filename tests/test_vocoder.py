from vocoder import mel_to_wave
from tests.test_mel import generate_random_wav, show_melspectrogram
import librosa
import torchaudio
import numpy as np

if __name__ == '__main__':

    wavs = []
    sr = 22050

    for _ in range(3):
        y, sr = generate_random_wav()
        wavs.append(y)

    mels = []
    for y in wavs:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
        mels.append(mel)
   
    mels_batch = np.stack(mels)

    for i, mel in enumerate(mels):
        show_melspectrogram(mel, sr, title=f"Mel Spectrogram{i + 1}")
    
    waveforms = mel_to_wave(mels_batch)

    for i, waveform in enumerate(waveforms):

        if waveform.ndimension() == 1:
            waveform = waveform.unsqueeze(0)  
        torchaudio.save(f'waveform_{i + 1}.wav', waveform, sr)