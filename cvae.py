import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import para_test
# -----------------------------
# 配置
# -----------------------------
input_dim = 6       # 生成目标的维度
condition_dim = 3   # 条件维度
latent_dim = 32   # 潜在变量维度
hidden_dim = 32
batch_size = 64
epochs = 150
learning_rate = 1e-3
num_samples = 1690

data = pd.read_csv("output.csv").to_numpy()
C = np.concatenate([data[:,8:], data[:,6:7]], axis=1)
X = data[:,0:6]
Q = data[:,7:8]


dataset = TensorDataset(torch.Tensor(X), torch.Tensor(C), torch.Tensor(Q))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)




class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dim):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        ###### Encoder : 7 + 2 - > 62
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
        )

    def encode(self, x, c):

        h = self.encoder(torch.cat([x, c], dim=1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):

        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)

        x_recon = self.decode(z, c)
        return x_recon, mu, logvar

def loss_functon(recon_x, x, mu, logvar):

    recon_loss = F.mse_loss(recon_x, x)
    q_pre = (recon_x[1] * 2 * torch.pi ) / recon_x[0]

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld * 1.5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CVAE(input_dim, condition_dim, latent_dim, hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, c_batch, q_batch in dataloader:

        x_batch = x_batch.to(device)
        c_batch = c_batch.to(device)
        q_batch = q_batch.to(device)
        optimizer.zero_grad()
        x_recon_batch, mu, logvar = model(x_batch, c_batch)
        loss = loss_functon(x_recon_batch, x_batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_samples:.4f}")

model.eval()
with ((torch.no_grad())):
    targetR = 123.45
    targetL = 3.76
    f = 50.5
    condition = torch.tensor([[targetR, targetL, f]]).to(device)

    z = 2*torch.randn(600, latent_dim).to(device)

    condition = condition.expand(z.size(0), -1)


    generated = model.decode(z, condition)
    generated = generated.cpu().numpy()
    generated = np.column_stack([generated, np.full(generated.shape[0], f)])

    print("Generated 7D outputs:\n", generated)
    para_test.test(generated, targetR, targetL, f)

