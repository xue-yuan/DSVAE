import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
import matplotlib.pyplot as plt

dataset = torchaudio.datasets.VCTK_092("./", download=True, audio_ext=".wav")
waveform, sample_rate, _, _ = dataset[0]

# waveform, sample_rate = torchaudio.load("./test.wav")
# mel_spectrogram = MelSpectrogram(
#     sample_rate=sample_rate,
# )
# mel_spec = mel_spectrogram(waveform)
# mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

# plt.figure(figsize=(10, 4))
# plt.imshow(mel_spec_db[1].numpy(), aspect="auto", origin="lower", cmap="viridis")
# plt.colorbar(format="%+2.0f dB")
# plt.xlabel("Time (frames)")
# plt.ylabel("Mel Frequency")
# plt.title("Mel Spectrogram")
# plt.show()

# plt.imshow(mel_spec[1])
# plt.show()
