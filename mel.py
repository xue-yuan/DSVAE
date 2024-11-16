import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

dataset = torchaudio.datasets.VCTK_092("./", download=True)
train_data_loader = DataLoader(dataset, batch_size=128, shuffle=True)


# s = set()

# print(len(dataset))

# for data in dataset:
#     waveform, sample_rate, _, _, _ = data
#     s.add(waveform.shape)

# d = []

# for ss in s:
#     d.append(ss[1])

# # print(s)

# print(min(d), max(d))

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
