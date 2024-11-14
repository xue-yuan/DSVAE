import torch
from torch import nn
from torchsummary import summary


class ShareBlock(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=5,
                padding=2,
                stride=1,
            ),
            nn.InstanceNorm1d(
                num_features=256,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class ShareEncoder(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.layers = nn.Sequential(
            ShareBlock(in_channels=in_channels),
            ShareBlock(in_channels=256),
            ShareBlock(in_channels=256),
        )

    def forward(self, x):
        return self.layers(x)


class SpeakerEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(
            in_features=1,
            out_features=64,
        )

    def forward(self, x):
        z1, _ = self.lstm(x)
        z2 = self.avg_pool(z1)
        z3 = self.fc(z2)

        return z3


class ContentEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.rnn = nn.RNN(
            input_size=1024,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(
            in_features=512,
            out_features=64,
        )

    def forward(self, x):
        z1, _ = self.lstm(x)
        z2, _ = self.rnn(z1)
        z3 = self.fc(z2)

        return z3


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.share_encoder = ShareEncoder()
        self.speaker_encoder = SpeakerEncoder()
        self.content_encoder = ContentEncoder()

    def forward(self, x):
        z = self.share_encoder(x)
        z_s = self.speaker_encoder(z)
        z_c = self.content_encoder(z)

        return z_s, z_c


class PreNetBlock(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.InstanceNorm2d(num_features=512),
            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=5,
                padding=2,
                stride=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class PostNetBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=5,
                padding=2,
                stride=1,
            ),
            nn.Tanh(),
            nn.InstanceNorm2d(
                num_features=512,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class PreNet(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.layers = nn.Sequential(
            PreNetBlock(
                num_features=num_features,
            ),
            PreNetBlock(
                num_features=num_features,
            ),
            PreNetBlock(
                num_features=num_features,
            ),
            nn.LSTM(
                input_size=512,
                hidden_size=512,
                num_layers=1,
                batch_first=True,
            ),
            nn.LSTM(
                input_size=512,
                hidden_size=1024,
                num_layers=2,
                batch_first=True,
            ),
            nn.Linear(in_features=1024, out_features=80),
        )

    def forward(self, x):
        return self.layers(x)


class PostNet(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.layers = nn.Sequential(
            PostNetBlock(
                in_channels=in_channels,
            ),
            PostNetBlock(
                in_channels=in_channels,
            ),
            PostNetBlock(
                in_channels=in_channels,
            ),
            PostNetBlock(
                in_channels=in_channels,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.layers = nn.Sequential(
            PreNet(
                num_features=num_features,
            ),
            PostNet(
                in_channels=num_features,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class Vocoder(nn.Module):

    def __init__(self):
        super().__init__()


class SpeakerPrior: ...


class ContentPrior: ...


class DSVAE(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.share_encoder = ShareEncoder(num_features)
        self.speaker_encoder = SpeakerEncoder()
        self.content_encoder = ContentEncoder()
        self.decoder = Decoder(num_features)
        self.vocoder = Vocoder()

    def concat(self, z_s, z_c):
        return z_s + z_c

    def forward(self, x):
        z = self.share_encoder(x)
        z_s = self.speaker_encoder(z)
        z_c = self.content_encoder(z)
        concat_x = self.concat(z_s, z_c)
        decoder_x = self.decoder(concat_x)
        spectrogram = self.vocoder(decoder_x)

        return spectrogram, z_s, z_c


def test_model(model, x):
    output = model(x)
    if isinstance(output, tuple):
        print(output[0].shape, output[1].shape)
    else:
        print(output.shape)


test_model(ShareEncoder(), torch.randn(30, 1, 512))
