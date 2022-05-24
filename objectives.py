import torch
from torch.distributions import kl_divergence as KL


def mmvae_elbo(model, x, K=1):
    """Computes importance-sampled m_elbo (in notes3) for multi-modal vae """
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = KL(qz_x, model.pz)
        # print(r, kld.shape)
        klds.append(kld.sum(-1))
        for d in range(len(px_zs)):
            lpx_z = px_zs[d][d].log_prob(x[d]).view(
                *px_zs[d][d].batch_shape[:2], -1)
            lpx_z = lpx_z.sum(-1)
            if d == r:
                lwt = torch.tensor(0.0)
            else:
                zs = zss[d].detach()
                lwt = (qz_x.log_prob(zs) -
                       qz_xs[d].log_prob(zs).detach()).sum(-1)

            lpx_zs.append(lwt.exp() * lpx_z)
    obj = (1 / 2) * \
        (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    return obj.mean(0).sum()


def pvae_elbo(model, x, k=1):
    zs, qz_xs, px_zs = model(x)
    z_mnist, z_svhn, z_poe = zs
    qz_x_mnist, qz_x_svhn, qz_x_poe = qz_xs
    px_z_mnist, px_z_svhn, px_z_mnist_poe, px_z_svhn_poe = px_zs

    klds = KL(qz_x_mnist, model.pz) + KL(qz_x_svhn,
                                         model.pz) + KL(qz_x_poe, model.pz)
    klds = klds.sum(axis=-1)

    lpx_z_mnist = px_z_mnist.log_prob(x[0]).view(
        *px_z_mnist.batch_shape[:2], -1)
    lpx_z_mnist = lpx_z_mnist.sum(axis=-1).squeeze()

    lpx_z_svhn = px_z_svhn.log_prob(x[1]).view(
        *px_z_svhn.batch_shape[:2], -1)
    lpx_z_svhn = lpx_z_svhn.sum(axis=-1).squeeze()

    lpx_z_mnist_poe = px_z_mnist_poe.log_prob(x[0]).view(
        *px_z_mnist_poe.batch_shape[:2], -1)
    lpx_z_mnist_poe = lpx_z_mnist_poe.sum(axis=-1).squeeze()

    lpx_z_svhn_poe = px_z_svhn_poe.log_prob(x[1]).view(
        *px_z_svhn_poe.batch_shape[:2], -1)
    lpx_z_svhn_poe = lpx_z_svhn_poe.sum(axis=-1).squeeze()

    return (lpx_z_mnist + lpx_z_svhn + lpx_z_mnist_poe + lpx_z_svhn_poe - klds).mean(0).sum()
