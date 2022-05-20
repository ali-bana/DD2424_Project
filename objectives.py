import torch
from funcs import kl_divergence
from pixyz.losses import KullbackLeibler
from pixyz.distributions import Laplace


def m_elbo(model, x, K=1):
    """Computes importance-sampled m_elbo (in notes3) for multi-modal vae """
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        # print(r, kld.shape)
        klds.append(kld.sum(-1))
        for d in range(len(px_zs)):
            lpx_z = px_zs[d][d].log_prob(x[d]).view(
                *px_zs[d][d].batch_shape[:2], -1)
            print(px_zs[d][d].log_prob(x[d]).shape, lpx_z.shape)
            lpx_z = (lpx_z * model.vaes[d].llik_scaling).sum(-1)
            if d == r:
                lwt = torch.tensor(0.0)
            else:
                zs = zss[d].detach()
                lwt = (qz_x.log_prob(zs) -
                       qz_xs[d].log_prob(zs).detach()).sum(-1)

            lpx_zs.append(lwt.exp() * lpx_z)
    print(len(lpx_zs))
    obj = (1 / len(model.vaes)) * \
        (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    return obj.mean(0).sum()
