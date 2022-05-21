from itertools import combinations
from turtle import forward
import torch
import torch.nn as nn
import os
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt, prod
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision.utils import save_image, make_grid
from vae_mnist import MNIST
from vae_svhn import SVHN
from torch.distributions import Normal
from vae_mnist import Enc_MNIST, Dec_MNIST
from vae_svhn import Enc_SVHN, Dec_SVHN


def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.mean
    except NotImplementedError:
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean


class My_MMVAE(nn.Module):
    def __init__(self, latent_dim) -> None:
        super(My_MMVAE, self).__init__()
        self.svhn_encoder = Enc_SVHN(latent_dim)
        self.svhn_decoder = Dec_SVHN(latent_dim)
        self.mnist_encoder = Enc_MNIST(latent_dim)
        self.mnist_decoder = Dec_MNIST(latent_dim)
        self.pz = Normal(0, 1)

    def forward(self, x):
        qz_x_mnist = Normal(*self.mnist_encoder(x[0]))
        qz_x_svhn = Normal(*self.svhn_encoder(x[1]))
        z_mnist = qz_x_mnist.rsample(torch.Size([1]))
        z_svhn = qz_x_svhn.rsample(torch.Size([1]))
        px_z_mnist = Normal(*self.mnist_decoder(z_mnist))
        px_z_svhn = Normal(*self.svhn_decoder(z_svhn))
        return [qz_x_mnist, qz_x_svhn], [[px_z_mnist, Normal(*self.svhn_decoder(z_mnist))], [Normal(*self.mnist_decoder(z_svhn)), px_z_svhn]], [z_mnist, z_svhn]

    def get_qz_x_mnist(self, x):
        return Normal(*self.mnist_encoder(x[0]))

    def get_qz_x_svhn(self, x):
        return Normal(*self.svhn_encoder(x[1]))

    def get_px_z_mnist(self, z):
        return Normal(*self.mnist_decoder(z))

    def get_px_z_svhn(self, z):
        return Normal(*self.svhn_decoder(z))


class MMVAE(nn.Module):
    def __init__(self, latent_dim):
        super(MMVAE, self).__init__()
        self.pz = Normal(0, 1)
        self.vaes = nn.ModuleList([MNIST(latent_dim), SVHN(latent_dim)])
        self.modelName = None  # filled-in per sub-class

    def forward(self, x, K=1):
        qz_xs, zss = [], []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))]
                 for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            qz_x, px_z, zs = vae(x[m], K=K)
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z  # fill-in diagonal
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    px_zs[e][d] = vae.px_z(*vae.dec(zs))
        return qz_xs, px_zs, zss

    def generate(self, N):
        self.eval()
        with torch.no_grad():
            data = []
            latents = self.pz.rsample(torch.Size([N]))
            for d, vae in enumerate(self.vaes):
                px_z = vae.px_z(*vae.dec(latents))
                data.append(px_z.mean.view(-1, *px_z.mean.size()[2:]))
        return data  # list of generations---one for each modality

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(data)
            # cross-modal matrix of reconstructions
            recons = [[get_mean(px_z) for px_z in r] for r in px_zs]
        return recons
