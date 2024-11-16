import sys

import torch
from torch.utils.data import DataLoader
from torchaudio import datasets, transforms

from models import DSVAE


train_data = datasets.VCTK_092("./", download=True)
train_data_loader = DataLoader(train_data, batch_size=128, shuffle=True)

model = DSVAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 100
losses = []

# for x, _ in train_loader:
#     print(x)


# for epoch in range(epochs):
#     model.train()
#     train_loss = 0

#     for x, _ in train_data_loader:
#         optimizer.zero_grad()

#         recon_x, mean_s, logvar_s, mean_c, logvar_c = model(x)
#         loss = loss_function(x, recon_x, mean_s, logvar_s, mean_c, logvar_c)
#         loss.backward()

#         train_loss += loss.item()
#         optimizer.step()

#     avg = train_loss / len(train_data_loader.dataset)
#     losses.append(avg)

#     print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_data_loader.dataset):.4f}")

# print("Done!")
