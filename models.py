import torch
from torch import nn


def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)

    return mean + epsilon * std


class ShareBlock(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=5,
                padding=2,
                stride=1,
            ),
            nn.InstanceNorm2d(
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
            input_size=32 * 256,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_mean = nn.Linear(
            in_features=1024,
            out_features=64,
        )
        self.fc_logvar = nn.Linear(
            in_features=1024,
            out_features=64,
        )

    def forward(self, x):
        z1, _ = self.lstm(x)
        z1 = z1.transpose(1, 2)
        z2 = self.avg_pool(z1)
        z2 = z2.squeeze(-1)
        mean = self.fc_mean(z2)
        logvar = self.fc_logvar(z2)

        return mean, logvar


class ContentEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=32 * 256,
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
        z2 = z2.mean(dim=1)
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
        z = z.permute(0, 3, 1, 2)
        z = z.reshape(z.shape[0], z.shape[1], -1)
        mean_s, logvar_s = self.speaker_encoder(z)
        mean_c, logvar_c = self.content_encoder(z)

        return mean_s, logvar_s, mean_c, logvar_c


class PreNetBlock(nn.Module):

    def __init__(self, in_channels=1):
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
        self.fc = nn.Linear(in_features=1024, out_features=128)

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
            PostNetBlock(),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.prenet = PreNet()
        self.postnet = PostNet()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=(1, 4),
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=(4, 4),
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, x):
        z = self.prenet(x)
        z = self.postnet(z)
        z = z.view(z.shape[0], z.shape[1], 1, 128)
        z = self.layers(z)
        return z


class Vocoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class DSVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.mean_s = 0
        self.logvar_s = 0
        self.mean_c = 0
        self.logvar_c = 0
        self.z_s = 0
        self.z_c = 0

    def concat(self, z_s, z_c):
        return torch.cat((z_s, z_c), dim=1)

    def forward(self, x):
        self.mean_s, self.logvar_s, self.mean_c, self.logvar_c = self.encoder(x)
        self.z_s = reparameterize(self.mean_s, self.logvar_s)
        self.z_c = reparameterize(self.mean_c, self.logvar_c)
        concat_x = self.concat(self.z_s, self.z_c)
        concat_x = concat_x.unsqueeze(1)
        z = self.decoder(concat_x)

        return z, self.mean_s, self.logvar_s, self.mean_c, self.logvar_c
