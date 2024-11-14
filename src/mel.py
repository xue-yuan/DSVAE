import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
import matplotlib.pyplot as plt

dataset = torchaudio.datasets.VCTK_092("./", download=True)

s = set()

print(len(dataset))

for data in dataset:
    waveform, sample_rate, _, _, _ = data
    s.add(waveform.shape)

print(s)

# waveform, sample_rate, _, _, _ = dataset[0]
# mel_spectrogram = MelSpectrogram(
#     sample_rate=sample_rate,
# )
# mel_spec = mel_spectrogram(waveform)
# mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
# print(waveform.shape)

# plt.figure(figsize=(10, 4))
# plt.imshow(mel_spec_db[0].numpy(), aspect="auto", origin="lower", cmap="viridis")
# plt.colorbar(format="%+2.0f dB")
# plt.xlabel("Time (frames)")
# plt.ylabel("Mel Frequency")
# plt.title("Mel Spectrogram")
# plt.show()
