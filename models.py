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
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=(4, 4),
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
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

    def concat(self, z_s, z_c):
        return torch.cat((z_s, z_c), dim=1)

    def forward(self, x):
        mean_s, logvar_s, mean_c, logvar_c = self.encoder(x)
        z_s = reparameterize(mean_s, logvar_s)
        z_c = reparameterize(mean_c, logvar_c)
        concat_x = self.concat(z_s, z_c)
        concat_x = concat_x.unsqueeze(1)
        z = self.decoder(concat_x)

        return z, mean_s, logvar_s, mean_c, logvar_c


def test_model(model, x):
    output = model(x)
    if isinstance(output, tuple):
        outputs = []
        for o in output:
            outputs.append(o.shape)

        return outputs
    else:
        return output.shape


batch_size = 2
input_channel = 1
input_height = 32
input_width = 4096


def test_share_encoder():
    o = test_model(
        ShareEncoder(),
        torch.randn(batch_size, input_channel, input_height, input_width),
    )

    print(o)

    assert o[0] == batch_size
    assert o[1] == 256
    assert o[2] == input_height
    assert o[3] == input_width


def test_speaker_encoder():
    z = torch.randn(batch_size, 256, input_height, input_width)
    z = z.permute(0, 3, 1, 2)
    z = z.reshape(z.shape[0], z.shape[1], -1)
    o = test_model(
        SpeakerEncoder(),
        z,
    )

    print(o)

    assert o[0][0] == batch_size
    assert o[0][1] == 64
    assert o[1][0] == batch_size
    assert o[1][1] == 64


def test_content_encoder():
    z = torch.randn(batch_size, 256, input_height, input_width)
    z = z.permute(0, 3, 1, 2)
    z = z.reshape(z.shape[0], z.shape[1], -1)
    o = test_model(
        ContentEncoder(),
        z,
    )

    print(o)

    assert o[0][0] == batch_size
    assert o[0][1] == 64
    assert o[1][0] == batch_size
    assert o[1][1] == 64


def test_encoder():
    o = test_model(
        Encoder(),
        torch.randn(batch_size, input_channel, input_height, input_width),
    )

    print(o)

    assert o[0][0] == batch_size
    assert o[0][1] == 64
    assert o[1][0] == batch_size
    assert o[1][1] == 64
    assert o[2][0] == batch_size
    assert o[2][1] == 64
    assert o[3][0] == batch_size
    assert o[3][1] == 64


def test_pre_net_block():
    o = test_model(
        PreNetBlock(),
        torch.randn(batch_size, input_channel, 128),
    )

    print(o)

    assert o[0] == batch_size
    assert o[1] == 512
    assert o[2] == 128


def test_pre_net():
    o = test_model(
        PreNet(),
        torch.randn(batch_size, input_channel, 128),
    )

    print(o)

    assert o[0] == batch_size
    assert o[1] == 512
    assert o[2] == 128


def test_post_net_block():
    o = test_model(
        PostNetBlock(),
        torch.randn(batch_size, 512, 128),
    )

    print(o)

    assert o[0] == batch_size
    assert o[1] == 512
    assert o[2] == 128


def test_post_net():
    o = test_model(
        PostNet(),
        torch.randn(batch_size, 512, 128),
    )

    print(o)

    assert o[0] == batch_size
    assert o[1] == 512
    assert o[2] == 128


def test_decoder():
    o = test_model(
        Decoder(),
        torch.randn(batch_size, input_channel, 128),
    )

    print(o)

    assert o[0] == batch_size
    assert o[1] == input_channel
    assert o[2] == input_height
    assert o[3] == input_width


def test_dsvae():
    o = test_model(
        DSVAE(),
        torch.randn(batch_size, input_channel, input_height, input_width),
    )

    print(o)

    assert o[0][0] == batch_size
    assert o[0][1] == input_channel
    assert o[0][2] == input_height
    assert o[0][3] == input_width

    assert o[1][0] == batch_size
    assert o[1][1] == 64
    assert o[2][0] == batch_size
    assert o[2][1] == 64
    assert o[3][0] == batch_size
    assert o[3][1] == 64
    assert o[4][0] == batch_size
    assert o[4][1] == 64


# test_share_encoder()
# test_speaker_encoder()
# test_content_encoder()
# test_encoder()
# test_pre_net_block()
# test_pre_net()
# test_post_net_block()
# test_post_net()
# test_decoder()
test_dsvae()
