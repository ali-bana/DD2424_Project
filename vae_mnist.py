import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod, sqrt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from vae import VAE

# Constants
dataSize = torch.Size([1, 28, 28])
data_dim = int(prod(dataSize))
hidden_dim = 400


def extra_hidden_layer():
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))


# Classes
class Enc_MNIST(nn.Module):
    """ Generate latent parameters for MNIST image data. """

    def __init__(self, latent_dim):
        super(Enc_MNIST, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Linear(data_dim, hidden_dim), nn.ReLU(True)))
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-3], -1))  # flatten data
        lv = self.fc22(e)
        return self.fc21(e), torch.exp(lv)


class Dec_MNIST(nn.Module):
    """ Generate an MNIST image given a sample from the latent space. """

    def __init__(self, latent_dim):
        super(Dec_MNIST, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(True)))
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        p = self.fc3(self.dec(z))
        d = torch.sigmoid(p.view(*z.size()[:-1], *dataSize))  # reshape data
        d = d.clamp(1e-6, 1 - 1e-6)

        return d, torch.tensor(0.1).to(z.device)  # mean, length scale


class MNIST(VAE):
    """ Derive a specific sub-class of a VAE for MNIST. """

    def __init__(self, latent_dim):
        super(MNIST, self).__init__(
            Enc_MNIST(latent_dim),
            Dec_MNIST(latent_dim))
        self.modelName = 'mnist'
        self.dataSize = dataSize

    def generate(self, runPath, epoch):
        N, K = 64, 9
        samples = super(MNIST, self).generate(N, K).cpu()
        # wrangle things so they come out tiled
        samples = samples.view(K, N, *samples.size()
                               [1:]).transpose(0, 1)  # N x K x 1 x 28 x 28
        s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]
        save_image(torch.stack(s),
                   '{}/gen_samples_{:03d}.png'.format(runPath, epoch),
                   nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch):
        recon = super(MNIST, self).reconstruct(data[:8])
        comp = torch.cat([data[:8], recon]).data.cpu()
        save_image(comp, '{}/recon_{:03d}.png'.format(runPath, epoch))
