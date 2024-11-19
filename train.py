import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchaudio import datasets, transforms

from dataset import collate_fn_fixed
from models import DSVAE


resampler = transforms.Resample(orig_freq=48000, new_freq=16000)


def kld_with_any(mean_c, logvar_c, mean_wav, var_wav):
    return 0.5 * (
        torch.log(var_wav / logvar_c)
        + (logvar_c + (mean_c - mean_wav) ** 2) / var_wav
        - 1
    )


def kld_with_normal(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())


def recon_loss(x, recon_x):
    return F.mse_loss(recon_x, x)


def loss_function(x, recon_x, mean_s, logvar_s, mean_c, logvar_c):
    return (
        recon_loss(x, recon_x)
        + kld_with_normal(mean_s, logvar_s)
        + kld_with_any(mean_c, logvar_c)
    )


train_data = datasets.VCTK_092("./", download=True)
fixed_length = 794730
train_data_loader = DataLoader(
    train_data,
    batch_size=128,
    collate_fn=lambda batch: collate_fn_fixed(batch, fixed_length=fixed_length),
    shuffle=True,
)


model = DSVAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 100
losses = []


for epoch in range(epochs):
    model.train()
    train_loss = 0

    for x, _ in train_data_loader:
        x = x.unsqueeze(1)
        x_resampled = resampler(x)
        optimizer.zero_grad()

        recon_x, mean_s, logvar_s, mean_c, logvar_c = model(x_resampled)
        loss = loss_function(x_resampled, recon_x, mean_s, logvar_s, mean_c, logvar_c)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

    avg = train_loss / len(train_data_loader.dataset)
    losses.append(avg)

    print(f"Epoch {epoch+1}, Loss: {avg:.4f}")

print("Done!")
