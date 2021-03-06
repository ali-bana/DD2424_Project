import torch.nn as nn
import torch
from funcs import get_mean, kl_divergence
from torch.distributions import Normal


class VAE(nn.Module):
    def __init__(self, enc, dec):
        super(VAE, self).__init__()
        self.pz = Normal(0, 1)
        self.px_z = Normal
        self.qz_x = Normal
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self._qz_x_params = None  # populated in `forward`

    @property
    def qz_x_params(self):
        if self._qz_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qz_x_params

    def forward(self, x, K=1):
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    def generate(self, N, K):
        self.eval()
        with torch.no_grad():
            latents = self.pz.rsample(torch.Size([N]))
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
