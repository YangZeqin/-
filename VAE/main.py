"""
learn: row 77，torch.empty_like(x)初始化的变量和x的device相同
problem: 为什么要使用BCE Loss？感觉MSE Loss更加合适？
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

# 加载数据
def load_data(batch_size):
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    test_dataset = MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

# Encoder类
class Encoder(nn.Module):
    def __init__(self, c, latent_dims):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)   # out: c * 14 * 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=2*c, kernel_size=4, stride=2, padding=1) # out: 2c * 7 * 7
        self.fc_mu = nn.Linear(in_features=2*c*7*7, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=2*c*7*7, out_features=latent_dims)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Decoder类
class Decoder(nn.Module):
    def __init__(self, c, latent_dims):
        super(Decoder, self).__init__()
        self.c = c
        self.fc = nn.Linear(in_features=latent_dims, out_features=2*c*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=2*c, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(x.shape[0], 2*self.c, 7, 7)
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))
        return x

# VAE类
class VariationalAutoEncoder(nn.Module):
    def __init__(self, c, latent_dims):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder(c, latent_dims)
        self.decoder = Decoder(c, latent_dims)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def reparameterize(self, mu, logvar):
        var = torch.exp(logvar / 2)
        # eps = torch.randn(size=var.shape)    # device: CPU
        eps = torch.empty_like(var).normal_()  # device: GPU
        return mu + var * eps

# 损失函数
def loss_fn(x, recon_x, mu, logvar):
    recon_loss = F.mse_loss(recon_x.view(-1, 784), x.view(-1, 784), reduction="sum")
    # recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

    kl_loss = -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
    return recon_loss + kl_loss

# 训练模型
def train(num_epochs, train_dataloader, device, VAE, optimizer):
    train_loss = []
    print("Training...")
    for epoch in range(num_epochs):
        train_loss.append(0)
        num_batches = 0
        for x, _ in train_dataloader:
            x = x.to(device)
            recon_x, mu, logvar = VAE(x)
            loss = loss_fn(x, recon_x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss[-1] += loss.item()
            num_batches += 1

        train_loss[-1] /= num_batches
        print("Epoch:[%d / %d] Average Reconstruction Error : %f" % (epoch+1, num_epochs, train_loss[-1]))

# 绘制二维空间的图
def show_2d_latent_space(vae):
    def to_img(x):
        x = x.clamp(0, 1)
        return x

    def show_image(img):
        img = to_img(img)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    with torch.no_grad():
        latent_x = np.linspace(-1.5, 1.5, 20)
        latent_y = np.linspace(-1.5, 1.5, 20)
        latents = torch.FloatTensor(len(latent_y), len(latent_x), 2)
        for i, lx in enumerate(latent_x):
            for j, ly in enumerate(latent_y):
                latents[j, i, 0] = lx
                latents[j, i, 1] = ly
        latents = latents.view(-1, 2)

        latents = latents.to(device)
        image_recon = vae.decoder(latents)
        image_recon = image_recon.cpu()

        fig, ax = plt.subplots(figsize=(10, 10))
        show_image(torchvision.utils.make_grid(image_recon.data[:400], 20, 5))
        plt.show()


if __name__ == "__main__":
    batch_size = 128
    channels = 64
    latent_dims = 2
    learning_rate = 1e-3
    num_epochs = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader, test_dataloader = load_data(batch_size)

    VAE = VariationalAutoEncoder(channels, latent_dims)
    VAE = VAE.to(device)

    optimizer = torch.optim.Adam(VAE.parameters(), lr=learning_rate, weight_decay=1e-5)

    train(num_epochs, train_dataloader, device, VAE, optimizer)

    torch.save(VAE.state_dict(), "VAE_epochs20_MSE.pth")

    vae = VariationalAutoEncoder(channels, latent_dims)
    vae = vae.to(device)
    vae.load_state_dict(torch.load("VAE_epochs20_MSE.pth"))

    show_2d_latent_space(vae)

    print("finish")