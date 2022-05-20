from mnist_svhn import get_data_loader
from mmvae import MNIST_SVHN
from torch.utils.data import dataloader
import torch
from objectives import m_elbo


class params:
    latent_dim = 20
    num_hidden_layers = 3
    learn_prior = False
    llik_scaling = False


def split_data_label(data):
    return [(d[0][0], d[1][0]) for d in data], [(d[0][1], d[1][1]) for d in data]


if __name__ == '__main__':
    model = MNIST_SVHN(20)
    print(type(model))
    train, test = get_data_loader(2)

    for d in train:
        data = [d[0][0], d[1][0]]
        print(torch.max(d[0][0]))
        print(torch.max(d[1][0]))
        loss = -m_elbo(model, data, 1)
        # print(loss)
        break
