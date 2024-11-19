import torchaudio

from prior import Wav2Vec

dataset = torchaudio.datasets.VCTK_092("./", download=True)
waveform, sample_rate, _, _, _ = dataset[0]

resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform_16k = resampler(waveform)

wav = Wav2Vec()
mean, variance = wav.get_mean_and_variance(waveform_16k)
print(mean, variance)
print(len(mean), len(variance))
