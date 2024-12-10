import torch
from torch import nn
from models import reparameterize, test_model

LENGTH = 794730


class StudentShareEncoder(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=5, padding=2, stride=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2, stride=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class StudentSpeakerEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=LENGTH, hidden_size=256, num_layers=1, batch_first=True
        )
        self.fc_mean = nn.Linear(256, 30)
        self.fc_logvar = nn.Linear(256, 30)

    def forward(self, x):
        z1, _ = self.lstm(x)
        z1 = z1.mean(dim=1)
        mean = self.fc_mean(z1)
        logvar = self.fc_logvar(z1)
        return mean, logvar


class StudentContentEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=LENGTH, hidden_size=256, num_layers=1, batch_first=True
        )
        self.fc_mean = nn.Linear(256, 30)
        self.fc_logvar = nn.Linear(256, 30)

    def forward(self, x):
        z1, _ = self.lstm(x)
        z1 = z1.mean(dim=1)
        mean = self.fc_mean(z1)
        logvar = self.fc_logvar(z1)
        return mean, logvar


class StudentEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.share_encoder = StudentShareEncoder()
        self.speaker_encoder = StudentSpeakerEncoder()
        self.content_encoder = StudentContentEncoder()

    def forward(self, x):
        z = self.share_encoder(x)
        mean_s, logvar_s = self.speaker_encoder(z)
        mean_c, logvar_c = self.content_encoder(z)
        return mean_s, logvar_s, mean_c, logvar_c


class StudentDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=60, hidden_size=512, num_layers=1, batch_first=True
        )
        self.fc = nn.Linear(512, 80)

    def forward(self, x):
        z, _ = self.lstm(x)
        z = self.fc(z)
        return z


class StudentDSVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = StudentEncoder()
        self.decoder = StudentDecoder()

    def concat(self, z_s, z_c):
        return torch.cat((z_s, z_c), dim=2)

    def forward(self, x):
        mean_s, logvar_s, mean_c, logvar_c = self.encoder(x)
        z_s = reparameterize(mean_s, logvar_s)
        z_c = reparameterize(mean_c, logvar_c)
        concat_x = self.concat(z_s, z_c)
        z = self.decoder(concat_x)
        return z, mean_s, logvar_s, mean_c, logvar_c


test_model(StudentEncoder(), torch.randn(1, 1, LENGTH))
# test_model(StudentContentEncoder(), torch.randn(1, 128, LENGTH))
