import torch
import numpy as np
import matplotlib.pyplot as plt
from mnist_svhn import get_data_loader


def display_flat_image(reshaped):
    reshaped += -1 * reshaped.min()
    reshaped /= reshaped.max()
    r = reshaped[0, :, :]
    g = reshaped[1, :, :]
    b = reshaped[2, :, :]
    r -= r.min()
    g -= g.min()
    b -= g.min()
    r /= r.max()
    g /= g.max()
    b /= b.max()
    plt.imshow(np.dstack((r, g, b)))
    plt.show()


def display_firsts_mmvae(model, loader):
    for d in loader:
        data = [d[0][0], d[1][0]]
        qz_xs, px_zs, zss = model(data)
        generated = px_zs[0][0].mean[0][0][0].detach().numpy()
        plt.imshow(d[0][0][0][0].numpy())
        plt.show()
        plt.imshow(generated)
        plt.show()
        # isplay svhn
        display_flat_image(d[1][0][0])
        generated = px_zs[1][1].mean[0][0].detach().numpy()
        display_flat_image(generated)
        # print(px_zs[1][1].mean.shape)


def display_firsts_pvae(model, loader):
    for d in loader:
        data = [d[0][0], d[1][0]]
        zs, qz_xs, px_zs = model(data)
        z_mnist, z_svhn, z_poe = zs
        qz_x_mnist, qz_x_svhn, qz_x_poe = qz_xs
        px_z_mnist, px_z_svhn, px_z_mnist_poe, px_z_svhn_poe = px_zs
        data = [d[0][0], d[1][0]]
        generated = px_z_mnist.mean[0][0][0].detach().numpy()
        plt.imshow(d[0][0][0][0].numpy())
        plt.show()
        plt.imshow(generated)
        plt.show()
        generated = px_z_mnist_poe.mean[0][0][0].detach().numpy()
        plt.imshow(generated)
        plt.show()
        # isplay svhn
        display_flat_image(d[1][0][0])
        generated = px_z_svhn.mean[0][0].detach().numpy()
        display_flat_image(generated)
        generated = px_z_svhn_poe.mean[0][0].detach().numpy()
        display_flat_image(generated)
        # print(px_zs[1][1].mean.shape)


if __name__ == '__main__':
    train_loader, test_loader, val_loader = get_data_loader(2, 'data/')

    # model = torch.load('saves/model_trying', map_location=torch.device('cpu'))
    # display_firsts_mmvae(model, train_loader)

    model = torch.load('saves/final_pvae', map_location=torch.device('cpu'))
    display_firsts_pvae(model, train_loader)
