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
        self.fc_mean = nn.Linear(
            in_features=1,
            out_features=64,
        )
        self.fc_logvar = nn.Linear(
            in_features=1,
            out_features=64,
        )

    def forward(self, x):
        z1, _ = self.lstm(x)
        z2 = self.avg_pool(z1)
        mean = self.fc_mean(z2)
        logvar = self.fc_logvar(z2)

        return mean, logvar


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
        self.fc_mean = nn.Linear(
            in_features=512,
            out_features=64,
        )
        self.fc_logvar = nn.Linear(
            in_features=512,
            out_features=64,
        )

    def forward(self, x):
        z1, _ = self.lstm(x)
        z2, _ = self.rnn(z1)
        mean = self.fc_mean(z2)
        logvar = self.fc_logvar(z2)

        return mean, logvar


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.share_encoder = ShareEncoder()
        self.speaker_encoder = SpeakerEncoder()
        self.content_encoder = ContentEncoder()

    def forward(self, x):
        z = self.share_encoder(x)
        mean_s, logvar_s = self.speaker_encoder(z)
        mean_c, logvar_c = self.content_encoder(z)

        return self.reparameterize(mean_s, logvar_s), self.reparameterize(
            mean_c, logvar_c
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)

        return mean + eps * std


class PreNetBlock(nn.Module):

    def __init__(self, in_channels=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.InstanceNorm1d(num_features=in_channels),
            nn.Conv1d(
                in_channels=in_channels,
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

    def __init__(self, out_channels=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=512,
                out_channels=out_channels,
                kernel_size=5,
                padding=2,
                stride=1,
            ),
            nn.Tanh(),
            nn.InstanceNorm1d(
                num_features=out_channels,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class PreNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            PreNetBlock(),
            PreNetBlock(
                in_channels=512,
            ),
            PreNetBlock(
                in_channels=512,
            ),
        )
        self.lstm1 = nn.LSTM(
            input_size=128,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=512,
            hidden_size=1024,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(in_features=1024, out_features=80)

    def forward(self, x):
        z1 = self.layers(x)
        z2, _ = self.lstm1(z1)
        z3, _ = self.lstm2(z2)
        z4 = self.fc(z3)

        return z4


class PostNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            PostNetBlock(),
            PostNetBlock(),
            PostNetBlock(),
            PostNetBlock(1),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            PreNet(),
            PostNet(),
        )

    def forward(self, x):
        return self.layers(x)


class Vocoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SpeakerPrior: ...


class ContentPrior: ...


class DSVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.vocoder = Vocoder()

    def concat(self, z_s, z_c):
        return torch.cat((z_s, z_c), dim=2)

    def forward(self, x):
        z_s, z_c = self.encoder(x)
        concat_x = self.concat(z_s, z_c)
        decoder_x = self.decoder(concat_x)
        # spectrogram = self.vocoder(decoder_x)

        return decoder_x, z_s, z_c


def test_model(model, x):
    output = model(x)
    if isinstance(output, tuple):
        for i in range(len(output)):
            print(f"{i}:", output[i].shape)
    else:
        print(output.shape)


# test_model(Encoder(), torch.randn(30, 1, 512)) # -> 30, 256, 64
# test_model(PreNet(), torch.randn(30, 256, 128))  # -> 30, 512, 80
# test_model(PostNet(), torch.randn(30, 512, 80))  # -> 30, 512, 80
# test_model(Decoder(), torch.randn(30, 256, 128)) # -> 30, 1, 80
test_model(DSVAE(), torch.randn(30, 1, 512))  # -> 30, 1, 80
