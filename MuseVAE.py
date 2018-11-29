import argparse
import numpy as np
import torch
import torch.utils.data
import nn
from codebase import utils as ut
from torch import nn, optim
from torch.nn import functional as F

class MuseVAE(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, batch_size, seq_length=1, gen_weight=1, class_weight=100):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.gen_weight = gen_weight
        self.class_weight = class_weight
        nn = getattr(nn, 'model')
        self.enc = nn.Encoder(self.x_dim, self.y_dim, self.z_dim, self.batch_size, self.seq_length)
        self.dec = nn.Encoder(self.x_dim, self.y_dim, self.z_dim, self.batch_size, self.seq_length)
        self.cls = nn.Classifier(self.x_dim, self.y_dim, self.batch_size, self.seq_length)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def nelbo_bound(self, x):
        pass

    def classification_loss(self, x, y):
        loss = nn.MSELoss()
        output = self.cls.classify(x)
        return loss(output, y)

    def loss(self, xl, yl):
        nelbo, kl_z, kl_y, rec = self.negative_elbo_bound(x)
        cl_loss = self.classification_loss(xl, yl)
        loss = self.gen_weight*nelbo + self.class_weight*cl_loss
        summaries = dict((
            ('train/loss', loss),
            ('class/ce', ce),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/kl_y', kl_y),
            ('gen/rec', rec),
        ))

        return loss, summaries
