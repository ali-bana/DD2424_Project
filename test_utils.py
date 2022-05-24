import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score


def get_latent(model, x, mode):
    if mode == 'mmvae':
        qz_xs, _, _ = model(x)
        qz_x_mnist = qz_xs[0]
        qz_x_svhn = qz_xs[1]
    elif mode == 'pvae':
        zs, qz_xs, px_zs = model(x)
        qz_x_mnist, qz_x_svhn, _ = qz_xs
    else:
        raise ValueError()

    return qz_x_mnist.mean.detach().numpy(), qz_x_svhn.mean.detach().numpy()


def get_accuracy(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    cm_argmax = cm.argmax(axis=0)
    cm_argmax
    y_pred_ = np.array([cm_argmax[i] for i in y_pred])
    cm_ = confusion_matrix(y, y_pred)
    return accuracy_score(y, y_pred_)


def get_latent_loader(model, loader, mode, max_n=np.inf):
    svhn = []
    mnist = []
    labels = []
    for data in tqdm(loader):
        labels.append(data[0][1].detach().numpy())
        d = [data[0][0], data[1][0]]
        mnist_mean, svhn_mean = get_latent(model, d, mode)
        mnist.append(mnist_mean)
        svhn.append(svhn_mean)
        if len(labels) * labels[-1].shape[0] >= max_n:
            break
    mnist = np.concatenate(mnist)
    svhn = np.concatenate(svhn)
    labels = np.concatenate(labels)
    return mnist, svhn, labels


def reshape_rgb(array):
    array += -1 * array.min()
    array /= array.max()
    r = array[0, :, :]
    g = array[1, :, :]
    b = array[2, :, :]
    r -= r.min()
    g -= g.min()
    b -= g.min()
    r /= r.max()
    g /= g.max()
    b /= b.max()
    return np.dstack((r, g, b))
