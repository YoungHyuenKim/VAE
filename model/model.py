import torch
import torch.nn as nn


class CVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, c_dim):
        super(CVAE, self).__init__()
        # encoder       part
        self.in_dim = x_dim
        self.fc1 = nn.Linear(x_dim + c_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim + c_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x, c):
        concat_input = torch.cat([x, c], 1)
        h = self.relu(self.fc1(concat_input))
        h = self.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps  # return z sample ( mu + std*eps)

    def decoder(self, z, c):
        concat_input = torch.cat([z, c], 1)
        h = self.relu(self.fc4(concat_input))
        h = self.relu(self.fc5(h))
        return self.sigmoid(self.fc6(h))

    def forward(self, x, c):
        mu, log_var = self.encoder(x.view(-1, self.in_dim), c)
        z = self.sampling(mu, log_var)
        return self.decoder(z, c), mu, log_var
