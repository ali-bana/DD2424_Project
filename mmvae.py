import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torch.distributions import Normal
from vae_mnist import Enc_MNIST, Dec_MNIST
from vae_svhn import Enc_SVHN, Dec_SVHN


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
