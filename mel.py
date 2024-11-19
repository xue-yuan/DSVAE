import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

dataset = torchaudio.datasets.VCTK_092("./", download=True)
train_data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
waveform, sample_rate, _, _, _ = dataset[0]

# print(sample_rate, waveform.shape[1])


def find_max_length(dataset):
    max_length = 0
    for i in range(len(dataset)):
        waveform, _, _, _, _ = dataset[i]
        max_length = max(max_length, waveform.shape[1])
    return max_length


# print(find_max_length(dataset))

s = set()

# print(len(dataset))

for data in dataset:
    waveform, sample_rate, _, _, _ = data

    s.add(waveform.shape[1] / sample_rate)

    # s.add(waveform.shape)

d = []

# for ss in s:
#     d.append(ss[1])

# print(s)

print(min(s), max(s))

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
