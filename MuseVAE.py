import argparse
import numpy as np
import torch
import torch.utils.data
import nns
from torch import nn, optim
from torch.nn import functional as F

def kl_normal(qm, qv, pm, pv):
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl

def sample_gaussian(m, v):
    epsilon = torch.randn_like(v)
    var = torch.mul(epsilon, torch.sqrt(v))
    return m + var

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
        nn = getattr(nns, 'model')
        self.enc = nn.Encoder(self.x_dim, self.y_dim, self.z_dim, self.batch_size, self.seq_length)
        self.dec = nn.Decoder(self.x_dim, self.y_dim, self.z_dim, self.batch_size, self.seq_length)
        self.cls = nn.Classifier(self.x_dim, self.y_dim, self.batch_size, self.seq_length)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def nelbo_bound(self, x, y):
        m, v = self.enc.encode(x, y)
        kl_z = kl_normal(m, v, self.z_prior_m, self.z_prior_v)

        z_samp = sample_gaussian(m, v)
        logits = self.dec.decode(z_samp, y)
        rec = nn.MSELoss()(x, logits)

        nelbo = rec + kl_z

        return nelbo, kl_z, rec
    def classification_loss(self, x, y):
        loss = nn.MSELoss()
        output = self.cls.classify(x)
        return loss(output, y)

    def loss(self, xl, yl):
        nelbo, kl_z, rec = self.nelbo_bound(xl, yl)
        cl_loss = self.classification_loss(xl, yl)
        loss = self.gen_weight*nelbo + self.class_weight*cl_loss
        summaries = dict((
            ('train/loss', loss),
            ('class/ce', cl_loss),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/rec', rec),
        ))

        return loss, summaries
