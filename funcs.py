import torch
import math


def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.mean
    except NotImplementedError:
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        print('using torch kl')
        return torch.distributions.kl_divergence(d1, d2)
    else:
        print('estimating')
        samples = d1.rsample(torch.Size([K]))
        print('not torch in kl')
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)
