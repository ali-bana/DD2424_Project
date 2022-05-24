from zmq import device
from mnist_svhn import get_data_loader
from mmvae import My_MMVAE
from MMVAE_they import MMVAE
from torch.utils.data import dataloader
import torch
from objectives import pvae_elbo, mmvae_elbo
from tqdm import tqdm
import numpy as np
from pvae import PVAE
torch.manual_seed(1377)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def split_data_label(data):
    return [(d[0][0], d[1][0]) for d in data], [(d[0][1], d[1][1]) for d in data]


def train(model: torch.nn.Module, data_loader: dataloader, objective, epochs, learning_rate, save_dir, validation_loader=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    print('using ' + device)
    for e in range(epochs):
        with tqdm(data_loader) as pbar:
            for d in pbar:
                data = [d[0][0].to(device), d[1][0].to(device)]
                loss = -objective(model, data, 1)
                loss.backward()
                optimizer.step()
                model.zero_grad()
                pbar.set_description(f'ELBO:{-loss:.3f}')
        torch.save(model, save_dir)
        if not validation_loader is None:
            val_elbo = []
            for d in validation_loader:
                data = [d[0][0].to(device), d[1][0].to(device)]
                val_elbo.append(-objective(model, data,
                                1).detach().cpu().numpy())
            print(f'epoch {e}/{epochs}, val_elbo:{np.mean(val_elbo):.3f}')

    return model


if __name__ == '__main__':
    # model1 = My_MMVAE(20)
    model = PVAE(20)
    # print(type(model))
    train_loader, test_loader, val_loader = get_data_loader(128, 'data/')
    # model = train(model1, train_loader, mmvae_elbo, 10,
    #               0.001, 'saves/mmvae', val_loader)
    model = train(model, train_loader, pvae_elbo, 10,
                  0.001, 'saves/mmvae', val_loader)
    # for d in train_loader:
    #     data = [d[0][0], d[1][0]]
    #     loss = -pvae_elbo(model, data)
    #     loss1 = -mmvae_elbo(model1, data)
    #     print(loss)
    #     print(loss1)
    #     break

    # #     # loss1 = -m_elbo(model, data, 1)
    # #     loss2 = -m_elbo(model1, data, 1)
    #     my_loss = -my_elbo(model1, data, 1).detach().numpy()
    #     print(np.mean([my_loss]))

    #     # print(loss1)
    #     print(loss2)
    #     print(my_loss)
    #     # print(loss)
    #     break
