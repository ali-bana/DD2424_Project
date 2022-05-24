import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torch.distributions import Normal
from vae_mnist import Enc_MNIST, Dec_MNIST
from vae_svhn import Enc_SVHN, Dec_SVHN


class PVAE(nn.Module):
    def __init__(self, latent_dim) -> None:
        super(PVAE, self).__init__()
        self.svhn_encoder = Enc_SVHN(latent_dim)
        self.svhn_decoder = Dec_SVHN(latent_dim)
        self.mnist_encoder = Enc_MNIST(latent_dim)
        self.mnist_decoder = Dec_MNIST(latent_dim)
        self.pz = Normal(0, 1)

    def forward(self, x):
        mu_mnist, var_mnist = self.mnist_encoder(x[0])
        qz_x_mnist = Normal(mu_mnist, var_mnist)
        mu_svhn, var_svhn = self.svhn_encoder(x[1])
        qz_x_svhn = Normal(mu_svhn, var_svhn)
        mu_p, var_p = self.POE(mu_mnist, var_mnist, mu_svhn, var_svhn)
        qz_x_poe = Normal(mu_p, var_p)

        z_mnist = qz_x_mnist.rsample(torch.Size([1]))
        z_svhn = qz_x_svhn.rsample(torch.Size([1]))
        z_poe = qz_x_poe.rsample(torch.Size([1]))

        px_z_mnist = Normal(*self.mnist_decoder(z_mnist))
        px_z_svhn = Normal(*self.svhn_decoder(z_svhn))

        px_z_mnist_poe = Normal(*self.mnist_decoder(z_poe))
        px_z_svhn_poe = Normal(*self.svhn_decoder(z_poe))

        zs = [z_mnist, z_svhn, z_poe]
        qz_xs = [qz_x_mnist, qz_x_svhn, qz_x_poe]
        px_zs = [px_z_mnist, px_z_svhn, px_z_mnist_poe, px_z_svhn_poe]

        return zs, qz_xs, px_zs

    def POE(self, mu_mnist, var_mnist, mu_svhn, var_svhn):
        t_mnist = 1 / var_mnist + 1e-8
        t_svhn = 1 / var_svhn + 1e-8
        mu_p = (mu_mnist*t_mnist + mu_svhn*t_svhn) / (t_mnist + t_svhn)
        var_p = 1 / (t_mnist + t_svhn)
        return mu_p, var_p

    def get_qz_x_mnist(self, x):
        return Normal(*self.mnist_encoder(x[0]))

    def get_qz_x_svhn(self, x):
        return Normal(*self.svhn_encoder(x[1]))

    def get_px_z_mnist(self, z):
        return Normal(*self.mnist_decoder(z))

    def get_px_z_svhn(self, z):
        return Normal(*self.svhn_decoder(z))
