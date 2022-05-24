from mnist_svhn import get_data_loader
import torch
from tqdm import tqdm
from torch.distributions import Normal
import matplotlib.pyplot as plt
from test_utils import reshape_rgb
from pre_trained_classifiers import mnist, svhn
import numpy as np


def generate_cross_modal(model, loader, mode, n_samples=1000):
    result = []
    for data in tqdm(loader):
        d = [data[0][0], data[1][0]]
        if mode == 'mmvae':
            qz_xs, _, _ = model(d)
            qz_x_mnist = qz_xs[0]
            qz_x_svhn = qz_xs[1]
        elif mode == 'pvae':
            zs, qz_xs, px_zs = model(d)
            qz_x_mnist, qz_x_svhn, _ = qz_xs

        z_svhn = qz_x_svhn.rsample(torch.Size([1]))
        print(z_svhn.shape)
        # z_svhn = qz_x_svhn.mean[None]
        # print(z_svhn.shape)
        px_zs = Normal(*model.mnist_decoder(z_svhn))
        generated_samples = px_zs.mean[0]
        result.append(generated_samples)
        if len(result)*result[-1].shape[0] >= n_samples:
            break
    return torch.concat(result)


def generate_cross_modal(model, loader, mode, n_samples):
    result_svhn = []
    result_mnist = []
    labels = []
    for data in tqdm(loader):
        d = [data[0][0], data[1][0]]
        if mode == 'mmvae':
            qz_xs, _, _ = model(d)
            qz_x_mnist = qz_xs[0]
            qz_x_svhn = qz_xs[1]
        elif mode == 'pvae':
            zs, qz_xs, px_zs = model(d)
            qz_x_mnist, qz_x_svhn, _ = qz_xs

        z_svhn = qz_x_svhn.rsample(torch.Size([1]))
        z_svhn = qz_x_svhn.mean[None]
        # z_svhn = qz_x_mnist.rsample(torch.Size([1]))
        px_zs = Normal(*model.mnist_decoder(z_svhn))
        generated_samples = px_zs.mean[0]
        result_mnist.append(generated_samples)

        z_mnist = qz_x_mnist.rsample(torch.Size([1]))
        z_mnist = qz_x_mnist.mean[None]
        # z_mnist = qz_x_svhn.rsample(torch.Size([1]))
        px_zs = Normal(*model.svhn_decoder(z_mnist))
        generated_samples = px_zs.mean[0]
        result_svhn.append(generated_samples)
        labels.append(data[0][1])
        if len(result_mnist)*result_mnist[-1].shape[0] >= n_samples:
            break
    return torch.concat(result_mnist), torch.concat(result_svhn), torch.concat(labels)


def display_and_save(img, save_file=None):
    assert len(img.shape) == 3
    if img.shape[0] == 1:
        img = img.squeeze().detach().numpy()
    elif img.shape[0] == 3:
        img = reshape_rgb(img.detach().numpy())
    else:
        raise ValueError()
    plt.imshow(img)
    if not save_file is None:
        plt.savefig(save_file)
    plt.close()


def pretrained_classify(data, dataset_name):
    if dataset_name == 'mnist':
        m = mnist()
        return torch.argmax(m(data), axis=1).numpy()
    elif dataset_name == 'svhn':
        m = svhn()
        data = data.detach().numpy()
        return np.argmax(m.predict(np.rollaxis(data, 1, 4)), axis=1)
    else:
        raise Exception()


def cross_modal_classification_task(model, loader, mode, n_samples=1000):
    mnist_g, svhn_g, labels = generate_cross_modal(
        model, loader, mode, n_samples)
    labels = labels.detach().numpy()
    pred_mnist = pretrained_classify(mnist_g, 'mnist')
    pred_svhn = pretrained_classify(svhn_g, 'svhn')
    mnist_acc = np.sum(pred_mnist == labels) / labels.shape[0]
    svhn_acc = np.sum(pred_svhn == labels) / labels.shape[0]
    return mnist_acc, svhn_acc


def axis_traverse_generator(model, loader, mode):
    result_mnist = []
    result_svhn = []
    for data in tqdm(loader):
        d = [data[0][0], data[1][0]]
        if mode == 'mmvae':
            qz_xs, _, _ = model(d)
            qz_x_mnist = qz_xs[0]
            qz_x_svhn = qz_xs[1]
        elif mode == 'pvae':
            zs, qz_xs, px_zs = model(d)
            qz_x_mnist, qz_x_svhn, _ = qz_xs
        z_mnist = qz_x_mnist.rsample([1])
        z_mnist = qz_x_mnist.mean[None]
        z_mnist = z_mnist.detach().numpy()
        z_svhn = qz_x_svhn.rsample([1])
        z_svhn = qz_x_svhn.mean[None]
        z_svhn = z_svhn.detach().numpy()
        for i in tqdm(range(20)):
            result_mnist.append([])
            result_svhn.append([])
            for t in range(-6, 7, 2):
                z_m = z_mnist.copy()
                z_m[:, :, i] += t
                z_m = torch.Tensor(z_m)
                px_z = Normal(*model.mnist_decoder(z_m))
                result_mnist[-1].append(px_z.mean)

                z_s = z_svhn.copy()
                z_s[:, :, i] += t
                z_s = torch.Tensor(z_s)
                px_z = Normal(*model.svhn_decoder(z_s))
                result_svhn[-1].append(px_z.mean)
        return result_mnist, result_svhn

        break


def axis_traverse_task(model, loader, mode):
    result_mnist, result_svhn = axis_traverse_generator(model, loader, mode)
    f, axarr = plt.subplots(len(result_mnist[0]), len(
        result_mnist), figsize=(5, 5))
    for i in range(20):
        for j in range(len(result_mnist[i])):
            axarr[j, i].imshow(result_mnist[i][j][0]
                               [0].squeeze().detach().numpy())
            axarr[j, i].axis('off')
            axarr[j, i].margins(x=0, y=0)
    plt.savefig(f'results/{mode}_mnist.png')
    plt.close()
    f, axarr = plt.subplots(len(result_mnist[0]), len(
        result_mnist), figsize=(5, 5))
    for i in range(20):
        for j in range(len(result_svhn[i])):
            axarr[j, i].imshow(reshape_rgb(
                result_svhn[i][j][0][0].detach().numpy()))
            axarr[j, i].axis('off')
            axarr[j, i].margins(x=0, y=0)
    plt.savefig(f'results/{mode}_svhn.png')
    plt.close()


if __name__ == '__main__':
    train_loader, test_loader, val_loader = get_data_loader(30, 'data/')
    # for data in val_loader:
    #     preds_mnist = pretrained_classify(data[0][0], 'mnist')
    #     preds_svhn = pretrained_classify(data[1][0], 'svhn')
    #     print(np.sum(preds_mnist == data[1][1].detach().numpy()))
    #     print(np.sum(preds_svhn == data[1][1].detach().numpy()))
    #     break

    model = torch.load('saves/model_trying', map_location=torch.device('cpu'))
    # print(cross_modal_classification_task(model, val_loader, 'mmvae', 1000))
    axis_traverse_task(model, val_loader, 'mmvae')

    # generated = generate_cross_modal(model, val_loader, 'mmvae', 100)
    # for i in range(10):
    #     print(generated[2][i])
    #     display_and_save(generated[0][i])
    #     display_and_save(generated[1][i])

    # print(generate_mnist(model, val_loader, 'mmvae').shape)
    # model = torch.load('saves/final_pvae', map_location=torch.device('cpu'))
    # generated = generate_mnist(model, val_loader, 'pvae', 100)
    # for i in range(10):
    #     display_and_save(generated[i])
    # m = mnist()
    # print(type(m))
    # m = svhn()
    # print(type(m))
