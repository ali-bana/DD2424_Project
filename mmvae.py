from itertools import combinations
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


class MMVAE(nn.Module):
    def __init__(self, prior_dist, latent_dim, *vaes):
        super(MMVAE, self).__init__()
        self.pz = prior_dist
        self.vaes = nn.ModuleList([vae(latent_dim) for vae in vaes])
        self.modelName = None  # filled-in per sub-class
        self._pz_params = None  # defined in subclass

    @property
    def pz_params(self):
        return self._pz_params

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

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
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
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


class MNIST_SVHN(MMVAE):
    def __init__(self, latent_dim):
        super(MNIST_SVHN, self).__init__(dist.Laplace, latent_dim, MNIST, SVHN)
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, latent_dim),
                         requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, latent_dim),
                         requires_grad=False)  # logvar
        ])
        self.modelName = 'mnist-svhn'

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)
