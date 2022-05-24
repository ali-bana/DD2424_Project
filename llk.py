import torch
from mnist_svhn import get_data_loader
import numpy as np
from tqdm import tqdm
from torch.distributions import Normal


def individual_llk(model, loader, mode, k=53):
    mnist_llks = []
    svhn_llks = []
    max_number = 500
    for data in tqdm(loader):
        d = [data[0][0], data[1][0]]
        if mode == 'mmvae':
            qz_xs, _, _ = model(d)
            qz_x_mnist = qz_xs[0]
            qz_x_svhn = qz_xs[1]
        elif mode == 'pvae':
            zs, qz_xs, px_zs = model(d)
            qz_x_mnist, qz_x_svhn, _ = qz_xs
        mnist = d[0]
        svhn = d[1]
        z_posterior_mnist = qz_x_mnist.rsample(torch.Size([k]))
        Px_mnist_z_mnist = Normal(
            *model.mnist_decoder(z_posterior_mnist)).log_prob(mnist).sum(-1).sum(-1).squeeze()
        Pz = Normal(0, 1).log_prob(z_posterior_mnist).sum(-1)
        Qz_x_mnist = qz_x_mnist.log_prob(z_posterior_mnist).sum(-1)
        llk_mnist = torch.logsumexp(
            Px_mnist_z_mnist+Pz-Qz_x_mnist-torch.log(torch.Tensor([k])).repeat(*Pz.shape), axis=0)
        mnist_llks.append(llk_mnist.detach().numpy())
        # SVHN
        z_posterior_svhn = qz_x_svhn.rsample(torch.Size([k]))
        Px_svhn_z_svhn = Normal(
            *model.svhn_decoder(z_posterior_svhn)).log_prob(svhn).sum(-1).sum(-1).sum(-1)
        Pz = Normal(0, 1).log_prob(z_posterior_svhn).sum(-1)
        Qz_x_svhn = qz_x_svhn.log_prob(z_posterior_svhn).sum(-1)
        llk_svhn = torch.logsumexp(
            Px_svhn_z_svhn+Pz-Qz_x_svhn-torch.log(torch.Tensor([k])).repeat(*Pz.shape), axis=0)
        svhn_llks.append(llk_svhn.detach().numpy())

        if len(svhn_llks) * svhn_llks[-1].shape[0] > max_number:
            break
    mnist_llks = np.concatenate(mnist_llks)
    svhn_llks = np.concatenate(svhn_llks)
    return mnist_llks.mean(), svhn_llks.mean()


def llk_one_give_other(model, loader, mode, k=53):
    mnist_llks = []
    svhn_llks = []
    max_number = 500
    for data in tqdm(loader):
        d = [data[0][0], data[1][0]]
        if mode == 'mmvae':
            qz_xs, _, _ = model(d)
            qz_x_mnist = qz_xs[0]
            qz_x_svhn = qz_xs[1]
        elif mode == 'pvae':
            zs, qz_xs, px_zs = model(d)
            qz_x_mnist, qz_x_svhn, _ = qz_xs
        mnist = d[0]
        svhn = d[1]
        # mnist|svhn
        z_svhn = qz_x_svhn.rsample(torch.Size([k]))
        P_mnist_z = Normal(*model.mnist_decoder(z_svhn)
                           ).log_prob(mnist).sum(-1).sum(-1).sum(-1)
        P_svhn_z = Normal(*model.svhn_decoder(z_svhn)
                          ).log_prob(svhn).sum(-1).sum(-1).sum(-1)
        Pz = Normal(0, 1).log_prob(z_svhn).sum(-1)
        Qz_svhn = qz_x_svhn.log_prob(z_svhn).sum(-1)
        first_term = torch.logsumexp(
            P_mnist_z+P_svhn_z+Pz-Qz_svhn-torch.log(torch.Tensor([k])).repeat(*Pz.shape), axis=0)
        z_prior = Normal(0, 1).rsample(z_svhn.shape)
        P_svhn_z_prior = Normal(
            *model.svhn_decoder(z_prior)).log_prob(svhn).sum(-1).sum(-1).sum(-1)
        second_term = torch.logsumexp(
            P_svhn_z_prior - torch.log(torch.Tensor([k])).repeat(*P_svhn_z_prior.shape), axis=0)

        mnist_llks.append((first_term - second_term).detach().numpy())

        # svhn|mnist
        z_mnist = qz_x_mnist.rsample(torch.Size([k]))
        P_svhn_z = Normal(*model.svhn_decoder(z_mnist)
                          ).log_prob(svhn).sum(-1).sum(-1).sum(-1)
        P_mnist_z = Normal(*model.mnist_decoder(z_mnist)
                           ).log_prob(mnist).sum(-1).sum(-1).sum(-1)
        Pz = Normal(0, 1).log_prob(z_mnist).sum(-1)
        Qz_mnist = qz_x_mnist.log_prob(z_mnist).sum(-1)
        first_term = torch.logsumexp(
            P_svhn_z+P_mnist_z+Pz-Qz_mnist-torch.log(torch.Tensor([k])).repeat(*Pz.shape), axis=0)
        z_prior = Normal(0, 1).rsample(z_mnist.shape)
        P_mnist_z_prior = Normal(
            *model.mnist_decoder(z_prior)).log_prob(mnist).sum(-1).sum(-1).sum(-1)
        second_term = torch.logsumexp(
            P_mnist_z_prior - torch.log(torch.Tensor([k])).repeat(*P_mnist_z_prior.shape), axis=0)

        svhn_llks.append((first_term - second_term).detach().numpy())

        if len(svhn_llks) * svhn_llks[-1].shape[0] > max_number:
            break
    mnist_llks = np.concatenate(mnist_llks)
    svhn_llks = np.concatenate(svhn_llks)
    return mnist_llks.mean(), svhn_llks.mean()


def llk_one_both(model, loader, mode, k=53):
    mnist_llks = []
    svhn_llks = []
    max_number = 500
    for data in tqdm(loader):
        d = [data[0][0], data[1][0]]
        if mode == 'mmvae':
            qz_xs, _, _ = model(d)
            qz_x_mnist = qz_xs[0]
            qz_x_svhn = qz_xs[1]
        elif mode == 'pvae':
            zs, qz_xs, px_zs = model(d)
            qz_x_mnist, qz_x_svhn, _ = qz_xs
        mnist = d[0]
        svhn = d[1]
        z_svhn = qz_x_svhn.rsample(torch.Size([k]))
        P_mnist_z = Normal(*model.mnist_decoder(z_svhn)
                           ).log_prob(mnist).sum(-1).sum(-1).sum(-1)
        P_svhn_z = Normal(*model.svhn_decoder(z_svhn)
                          ).log_prob(svhn).sum(-1).sum(-1).sum(-1)
        Pz = Normal(0, 1).log_prob(z_svhn).sum(-1)
        Qz_svhn = qz_x_svhn.log_prob(z_svhn).sum(-1)
        first_term = torch.logsumexp(
            P_mnist_z+P_svhn_z+Pz-Qz_svhn-torch.log(torch.Tensor([k])).repeat(*Pz.shape), axis=0)
        z_prior = Normal(0, 1).rsample(z_svhn.shape)
        P_svhn_z_prior = Normal(
            *model.svhn_decoder(z_prior)).log_prob(svhn).sum(-1).sum(-1).sum(-1)
        P_mnist_z_prior = Normal(
            *model.mnist_decoder(z_prior)).log_prob(mnist).sum(-1).sum(-1).sum(-1)
        second_term = torch.logsumexp(
            P_mnist_z_prior + P_svhn_z_prior - torch.log(torch.Tensor([k])).repeat(
                *P_svhn_z_prior.shape), axis=0)

        mnist_llks.append((first_term - second_term).detach().numpy())
        # svhn|mnist
        z_mnist = qz_x_mnist.rsample(torch.Size([k]))
        P_svhn_z = Normal(*model.svhn_decoder(z_mnist)
                          ).log_prob(svhn).sum(-1).sum(-1).sum(-1)
        P_mnist_z = Normal(*model.mnist_decoder(z_mnist)
                           ).log_prob(mnist).sum(-1).sum(-1).sum(-1)
        Pz = Normal(0, 1).log_prob(z_mnist).sum(-1)
        Qz_mnist = qz_x_mnist.log_prob(z_mnist).sum(-1)
        first_term = torch.logsumexp(
            P_svhn_z+P_mnist_z+Pz-Qz_mnist-torch.log(torch.Tensor([k])).repeat(*Pz.shape), axis=0)
        z_prior = Normal(0, 1).rsample(z_mnist.shape)
        P_mnist_z_prior = Normal(
            *model.mnist_decoder(z_prior)).log_prob(mnist).sum(-1).sum(-1).sum(-1)
        P_svhn_z_prior = Normal(
            *model.svhn_decoder(z_prior)).log_prob(svhn).sum(-1).sum(-1).sum(-1)
        second_term = torch.logsumexp(
            P_mnist_z_prior - torch.log(torch.Tensor([k])).repeat(*P_mnist_z_prior.shape), axis=0)

        svhn_llks.append((first_term - second_term).detach().numpy())

        if len(svhn_llks) * svhn_llks[-1].shape[0] > max_number:
            break
    mnist_llks = np.concatenate(mnist_llks)
    svhn_llks = np.concatenate(svhn_llks)
    return mnist_llks.mean(), svhn_llks.mean()


if __name__ == '__main__':
    train_loader, test_loader, val_loader = get_data_loader(25, 'data/')

    model = torch.load('saves/model_trying', map_location=torch.device('cpu'))
    # print(individual_llk(model, val_loader, 'mmvae'))
    print(llk_one_give_other(model, val_loader, 'mmvae'))
    print(llk_one_both(model, val_loader, 'mmvae'))
    model = torch.load('saves/final_pvae', map_location=torch.device('cpu'))
    # print(individual_llk(model, val_loader, 'pvae'))
    print(llk_one_give_other(model, val_loader, 'pvae'))
    print(llk_one_both(model, val_loader, 'pvae'))
