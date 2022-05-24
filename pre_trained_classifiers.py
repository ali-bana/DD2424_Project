import torch.nn as nn
from collections import OrderedDict
import torch
from mnist_svhn import get_data_loader
import keras
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model = nn.Sequential(layers)
        print(self.model)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)


def mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=None):
    model = MLP(input_dims, n_hiddens, n_class)

    m = torch.load('trained_classifier/mnist.pth',
                   map_location=torch.device('cpu'))
    state_dict = m.state_dict() if isinstance(m, nn.Module) else m
    model.load_state_dict(state_dict)
    return model


def svhn():
    return keras.models.load_model('trained_classifier/weights.hdf5')


if __name__ == '__main__':
    train_loader, test_loader, val_loader = get_data_loader(128, 'data/')
    m = svhn()
    for data in val_loader:
        d = data[1][0].detach().numpy()
        print(np.rollaxis(d, 1, 4).shape)
        preds = np.argmax(m.predict(np.rollaxis(d, 1, 4)), axis=1)
        print(preds == data[1][1].detach().numpy())
        break
    #     labels = torch.argmax(m(d), axis=1).detach().numpy()
    #     # print(type(labels))
    #     print(labels == data[1][1].detach().numpy())
    #     break
