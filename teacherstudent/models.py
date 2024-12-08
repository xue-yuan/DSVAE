import torch
from torch import nn


def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)

    return mean + epsilon * std


class StudentEncoder1(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=256,
                kernel_size=5,
                padding=2,
                stride=1,
            ),
            nn.InstanceNorm2d(
                num_features=256,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
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

        self.speaker_lstm = nn.LSTM(
            input_size=32 * 256,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.speaker_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.speaker_fc_mean = nn.Linear(
            in_features=1024,
            out_features=64,
        )
        self.speaker_fc_logvar = nn.Linear(
            in_features=1024,
            out_features=64,
        )

        self.content_lstm = nn.LSTM(
            input_size=32 * 256,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.content_rnn = nn.RNN(
            input_size=1024,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
        )
        self.content_fc_mean = nn.Linear(
            in_features=512,
            out_features=64,
        )
        self.content_fc_logvar = nn.Linear(
            in_features=512,
            out_features=64,
        )

    def forward(self, x):
        z = self.layers(x)
        z = z.permute(0, 3, 1, 2)
        z = z.reshape(z.shape[0], z.shape[1], -1)

        speaker_z1, _ = self.speaker_lstm(z)
        speaker_z1 = speaker_z1.transpose(1, 2)
        speaker_z2 = self.speaker_avg_pool(speaker_z1)
        speaker_z2 = speaker_z2.squeeze(-1)
        speaker_mean = self.speaker_fc_mean(speaker_z2)
        speaker_logvar = self.speaker_fc_logvar(speaker_z2)

        content_z1, _ = self.content_lstm(z)
        content_z2, _ = self.content_rnn(content_z1)
        content_z2 = content_z2.mean(dim=1)
        content_mean = self.content_fc_mean(content_z2)
        content_logvar = self.content_fc_logvar(content_z2)

        return speaker_mean, speaker_logvar, content_mean, content_logvar


class StudentDecoder1(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 128 * 16 * 2048)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 16, 2048)
        return self.layers(x)


class StudentDSVAE1(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = StudentEncoder1()
        self.decoder = StudentDecoder1()
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


class StudentEncoder2(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32 * 4096, 1024)
        self.fc_mean = nn.Linear(1024, 64)
        self.fc_logvar = nn.Linear(1024, 64)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        h = torch.relu(self.fc1(x_flat))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar, mean, logvar


class StudentDecoder2(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 32 * 4096)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        x_recon = self.fc2(h)
        return x_recon.view(-1, 1, 32, 4096)


class StudentDSVAE2(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = StudentEncoder2()
        self.decoder = StudentDecoder2()
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
        z = self.decoder(concat_x)

        return z, self.mean_s, self.logvar_s, self.mean_c, self.logvar_c
