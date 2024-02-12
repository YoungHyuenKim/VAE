import os
import glob
import numpy as np
import torch
import torchvision.datasets as dataset

import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from model.model import CVAE
import torch.nn.functional as F
from torchvision.utils import save_image

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
])
download_root = './MNIST_DATASET'

train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
valid_dataset = MNIST(download_root, transform=mnist_transform, download=True)
test_dataset = MNIST(download_root, transform=mnist_transform, download=True)


def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    in_dim = 28 * 28
    out_dim = 2
    h1 = 512
    h2 = 256

    condition_dim = 10 

    model = CVAE(in_dim, h1, h2, out_dim, condition_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 30

    test_interval = 10
    for epoch in range(epochs):
        for batch_idx, (x, target) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            c = F.one_hot(target, num_classes = condition_dim).float().to(device)
            recon_x, mu, log_var = model(x, c)
            loss = loss_function(recon_x, x, mu, log_var)

            loss.backward()
            optimizer.step()

        if epoch % test_interval == 0:
            x = np.linspace(-1, 1, 10)
            y = np.linspace(-1, 1, 10)
            # full coorindate arrays
            xx, yy = np.meshgrid(x, y)
            X = np.stack([xx, yy], axis=-1)
            rows, cols,_ = X.shape
            X = X.reshape((-1,2))
            n = X.shape[0]
            for cond in range(condition_dim):

                z = torch.tensor(X, dtype=torch.float32)
                c = torch.ones(n,dtype=torch.int) * cond
                c = F.one_hot(c.long(), num_classes=condition_dim)
                c = c.float().to(device)
                sample = model.decoder(z, c).to(device)

                save_image(sample.view(-1, 1, 28, 28), './samples/cvae_sample_' + f"{cond}_{epoch}" + '.png', nrow=10)
