import numpy as np
from mnist_svhn import get_data_loader
import scipy
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
from test_utils import get_latent_loader
plt.rcParams["figure.figsize"] = (8, 8)


def plot_latent(model, loader, mode, save_fig):
    max_points = 35000
    mnist, svhn, labels = get_latent_loader(model, loader, mode, max_points)
    length = mnist.shape[0]
    tsne = TSNE()
    transformed = tsne.fit_transform(np.concatenate([mnist, svhn]))
    plt.scatter(transformed[:length, 0],
                transformed[:length, 1], c=labels, s=2)
    plt.scatter(transformed[length:, 0],
                transformed[length:, 1], c=labels, marker='^', s=2)
    plt.legend()
    plt.savefig(save_fig, dpi=1200)
    plt.show()
    print(mnist.shape, svhn.shape, labels.shape)


if __name__ == '__main__':
    train_loader, test_loader, val_loader = get_data_loader(128, 'data/')

    # model = torch.load('saves/model_trying', map_location=torch.device('cpu'))
    # plot_latent(model, val_loader, 'mmvae', 'saves/mmvae_latent.png')
    model = torch.load('saves/final_pvae', map_location=torch.device('cpu'))
    plot_latent(model, val_loader, 'pvae', 'saves/pvae_latent.png')
