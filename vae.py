import torch.nn as nn
import torch
from funcs import get_mean, kl_divergence
from torch.distributions import Normal


class VAE(nn.Module):
    def __init__(self, prior_dist, likelihood_dist, post_dist, enc, dec):
        super(VAE, self).__init__()
        self.pz = Normal
        self.px_z = Normal
        self.qz_x = Normal
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self._pz_params = None  # defined in subclass
        self._qz_x_params = None  # populated in `forward`
        self.llik_scaling = 1.0

    @property
    def pz_params(self):
        return self._pz_params

    @property
    def qz_x_params(self):
        if self._qz_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qz_x_params

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, K=1):
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    def generate(self, N, K):
        self.eval()
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            px_z = self.px_z(*self.dec(latents))
            data = px_z.sample(torch.Size([K]))
        return data.view(-1, *data.size()[3:])

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            latents = qz_x.rsample()  # no dim expansion
            px_z = self.px_z(*self.dec(latents))
            recon = get_mean(px_z)
        return recon
