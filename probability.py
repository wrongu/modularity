import torch
from math import log


def log_normalize(logp):
    return logp - torch.logsumexp(logp.flatten(), dim=-1)


def log2prob(logp):
    return torch.exp(log_normalize(logp))


def temperature(logp, temp):
    return log_normalize(logp/temp)


def discrete_entropy(logp):
    logp = log_normalize(logp)
    plogp = -logp.exp()*logp
    plogp[torch.isnan(plogp)] = 0.
    plogp[torch.isinf(plogp)] = 0.
    plogp[plogp < 0.] = 0.
    return torch.sum(plogp)


def entropy_to_temperature(logp, target_entropy, init_t=1.0, min_temp=1e-6, max_temp=1e6, eps=0.001, max_steps=1000):
    log_t, step = torch.tensor(init_t).log(), 1.0
    min_log_t, max_log_t = log(min_temp), log(max_temp)
    ent = discrete_entropy(temperature(logp, log_t.exp()))
    for _ in range(max_steps):
        # Take a step towards the target (entropy is monotonic with temperature)
        new_log_t = log_t - step if ent > target_entropy else log_t + step
        new_ent = discrete_entropy(temperature(logp, torch.exp(new_log_t)))
        if abs(target_entropy - new_ent) < eps or new_log_t < min_log_t or new_log_t > max_log_t:
            break
        # If old value is closer than new value, we overshot... reduce step size and try again
        if abs(target_entropy - ent) < abs(target_entropy - new_ent):
            step /= 2
        else:
            log_t, ent = new_log_t, new_ent
    return min(max(torch.exp(new_log_t).item(), min_temp), max_temp)
