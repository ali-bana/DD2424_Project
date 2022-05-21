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


if __name__ == '__main__':
    model = torch.load('saves/model_trying', map_location=torch.device('cpu'))
    train_loader, test_loader, val_loader = get_data_loader(2, 'data/')
    for d in train_loader:
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
        break
