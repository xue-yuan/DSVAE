import torch
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
from torchaudio import datasets

from loader import collate_function
from models import DSVAE

has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")
torch.mps.empty_cache()


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
        # + kld_with_any(mean_c, logvar_c)
    )


train_data = datasets.VCTK_092("./", download=True)
train_data_loader = DataLoader(
    train_data,
    batch_size=4,
    collate_fn=lambda batch: collate_function(batch),
    sampler=RandomSampler(train_data, num_samples=100),
)

model = DSVAE()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 1
losses = []

torch.mps.empty_cache()
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for batch_idx, (x, _) in enumerate(train_data_loader):
        x = x.to(device)
        optimizer.zero_grad()

        recon_x, mean_s, logvar_s, mean_c, logvar_c = model(x)
        loss = loss_function(x, recon_x, mean_s, logvar_s, mean_c, logvar_c)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_data_loader)}")

    avg = train_loss / len(train_data_loader.dataset)
    losses.append(avg)

    print(f"Epoch {epoch+1}, Loss: {avg:.4f}")

print("Done!")
