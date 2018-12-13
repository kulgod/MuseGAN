import json
import numpy as np
from torch import autograd, nn, optim
from torch.nn import functional as F
import torch

class Encoder(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, batch_size, seq_length=1):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.net_1 = nn.GRU(self.x_dim+self.y_dim, self.z_dim)
        self.net_2 = nn.Linear(self.seq_length*self.z_dim, self.seq_length*self.z_dim)

    def encode(self, x, y):
        x = x.view(self.seq_length, self.batch_size, self.x_dim)
        y = y.unsqueeze(0).repeat(self.seq_length, 1, 1)
        xy = torch.cat((x, y), dim=-1)
        output, h_n = self.net_1(xy)
        output = output.contiguous().view(output.size(1), output.size(0)*output.size(2))
        z = self.net_2(output)
        return z.view(self.batch_size, self.seq_length*self.z_dim))


class Decoder(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, batch_size, seq_length=1):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.net_1 = nn.Linear(self.seq_length*self.z_dim+self.y_dim, self.seq_length*self.z_dim)
        self.net_2 = nn.GRU(self.z_dim, self.x_dim)
    def decode(self, z, y):
        z = z.view(self.batch_size, self.z_dim*self.seq_length)
        # y = y.repeat(self.seq_length, 1, 1)

        zy = torch.cat((z, y), dim=-1)
        h = self.net_1(zy)
        h = h.view(self.seq_length, self.batch_size, self.z_dim)
        x, x_n = self.net_2(h)
        x = x.view(self.batch_size, self.x_dim*self.seq_length)
        return x

class Classifier(nn.Module):
    def __init__(self, x_dim, y_dim, batch_size, seq_length=1):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.net = nn.Sequential(
            nn.Linear(self.seq_length*self.x_dim, self.seq_length*self.x_dim),
            nn.ReLU(),
            nn.Linear(self.seq_length*self.x_dim, self.y_dim),
            nn.Tanh()
        )

    def classify(self, x):
        x = x.view(self.batch_size, self.seq_length*self.x_dim)
        y = self.net(x)
        # m, h = torch.split(h, h.size(-1) // 2, dim=-1)
        # y_m = F.tanh(m)
        # y_v = F.softplus(h) + 1e-8
        return y
