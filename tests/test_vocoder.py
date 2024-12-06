from vocoder import mel_to_wave
from tests.test_mel import generate_random_wav, show_melspectrogram
import librosa
import torchaudio

if __name__ == '__main__':
    y1, sr1 = generate_random_wav()
    y2, sr2 = generate_random_wav()

    mel_1 = librosa.feature.melspectrogram(y=y1, sr=sr1, n_mels=128, fmax=8000)
    mel_2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128, fmax=8000)

    show_melspectrogram(mel_1, sr1, title="Mel Spectrogram 1")
    show_melspectrogram(mel_2, sr2, title="Mel Spectrogram 2")

    wavform_1 = mel_to_wave(mel_1)
    wavform_2= mel_to_wave(mel_2)
    
    torchaudio.save('waveform_1.wav', wavform_1.squeeze(1), 22050)
    torchaudio.save('waveform_2.wav', wavform_2.squeeze(1), 22050)
