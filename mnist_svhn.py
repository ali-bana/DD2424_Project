import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset, SplitDataset


def _rand_match_on_idx(l1, idx1, l2, idx2, max_d=10000, dm=10):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    _idx1, _idx2 = [], []
    for l in l1.unique():  # assuming both have same idxs
        l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
        n = min(l_idx1.size(0), l_idx2.size(0), max_d)
        l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
        for _ in range(dm):
            _idx1.append(l_idx1[torch.randperm(n)])
            _idx2.append(l_idx2[torch.randperm(n)])
    return torch.cat(_idx1), torch.cat(_idx2)


def get_data_loader(batch_size, data_folder, validation_frac=0.05):
    max_d = 10000  # maximum number of datapoints per class
    dm = 30        # data multiplier: random permutations to match

    # get the individual datasets
    tx = transforms.ToTensor()
    train_mnist = datasets.MNIST(
        data_folder, train=True, download=True, transform=tx)
    test_mnist = datasets.MNIST(
        data_folder, train=False, download=True, transform=tx)
    train_svhn = datasets.SVHN(
        data_folder, split='train', download=True, transform=tx)
    test_svhn = datasets.SVHN(data_folder, split='test',
                              download=True, transform=tx)
    # svhn labels need extra work
    train_svhn.labels = torch.LongTensor(
        train_svhn.labels.squeeze().astype(int)) % 10
    test_svhn.labels = torch.LongTensor(
        test_svhn.labels.squeeze().astype(int)) % 10

    mnist_l, mnist_li = train_mnist.targets.sort()
    svhn_l, svhn_li = train_svhn.labels.sort()
    idx_train_mnist, idx_train_svhn = _rand_match_on_idx(
        mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm)
    mnist_l, mnist_li = test_mnist.targets.sort()
    svhn_l, svhn_li = test_svhn.labels.sort()
    idx_test_mnist, idx_test_svhn = _rand_match_on_idx(
        mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm)
    train_loader = TensorDataset([
        ResampleDataset(train_mnist, lambda d,
                        i: idx_train_mnist[i], size=len(idx_train_mnist)),
        ResampleDataset(train_svhn, lambda d,
                        i: idx_train_svhn[i], size=len(idx_train_svhn))
    ])
    test_loader = TensorDataset([
        ResampleDataset(test_mnist, lambda d,
                        i: idx_test_mnist[i], size=len(idx_test_mnist)),
        ResampleDataset(test_svhn, lambda d,
                        i: idx_test_svhn[i], size=len(idx_test_svhn))
    ])

    val_loader = SplitDataset(
        train_loader, {'train': 1-validation_frac, 'val': validation_frac})
    val_loader.select('val')

    train_loader = SplitDataset(
        train_loader, {'train': 1-validation_frac, 'val': validation_frac})
    train_loader.select('train')

    # print(len(train_loader))
    # print(len(val_loader))
    # print(len(test_loader))
    train_loader = DataLoader(
        train_loader, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, val_loader


if __name__ == '__main__':
    train, test = get_data_loader()
    print(train[0][0][0].shape)
    print()
    print(train[0][1][0].shape)
