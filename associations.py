import torch
from torch.utils.data import DataLoader
from math import ceil
from tqdm import tqdm


# TODO - consider refactoring to use pl.Trainer.test() with callbacks


def corrcov(covariance, eps=1e-12):
    sigma = covariance.diag().sqrt()
    sigma[sigma < eps] = eps
    return covariance / (sigma.view(-1, 1) * sigma.view(1, -1))


def sum_hessian(loss, hidden):
    """Given scalar tensor 'loss' and [b, d] batch of hidden activations 'hidden', computes [d, d] size sum of hessians
    for all entries in the batch.

    We make use of the fact that grad^2(f) = grad(grad(f)) and that sum of grads = grad of sum. So, sum(grad^2(f)) is
    computed as grad(sum(grad(f))).
    """
    d = hidden.size(1)
    grad = torch.autograd.grad(loss, hidden, retain_graph=True, create_graph=True)[0]
    sum_grad = torch.sum(grad, dim=0)
    hessian = hidden.new_zeros(d, d)
    for i in range(d):
        hessian[i, :] = torch.autograd.grad(sum_grad[i], hidden, retain_graph=True)[0].sum(dim=0)
    return hessian


# TODO - include covariance in feature space (HSIC) for both fwd and bwd methods
# TODO - include 'online' methods
def get_associations(model, method, dataset, device='cpu', batch_size=200, shuffle=False):
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    num_batches = ceil(len(dataset)/batch_size)

    if method in ['forward_cov', 'forward_cov_norm']:
        moment1 = [torch.zeros(d, device=device) for d in model.hidden_dims]
        moment2 = [torch.zeros(d, d, device=device) for d in model.hidden_dims]
        for im, _ in tqdm(loader, desc=method, total=num_batches):
            hidden, _ = model(im.to(device))

            for h, m1, m2 in zip(hidden, moment1, moment2):
                m1 += h.sum(dim=0) / len(dataset)
                m2 += h.T @ h / len(dataset)

        # Convert from moment1 and moment2 to covariance for each hidden layer
        assoc = [m2 - m1.view(-1, 1) * m1.view(1, -1) for m1, m2 in zip(moment1, moment2)]

    elif method in ['backward_hess', 'backward_hess_norm']:
        assoc = [torch.zeros(d, d, device=device) for d in model.hidden_dims]
        for im, la in tqdm(loader, desc=method, total=num_batches):
            im, la = im.to(device), la.to(device)
            hidden, out = model(im)
            loss = model.loss_fn(im, la, out)
            for a, h in zip(assoc, hidden):
                a += sum_hessian(loss, h)
        # Ensure symmetry, since hessians will not be *exactly* symmetric up to floating point errors, and we will
        # be asserting symmetry later.
        assoc = [(a + a.T)/2. for a in assoc]
    else:
        allowed_methods = ['forward_cov', 'forward_cov_norm', 'backward_hess', 'backward_hess_norm']
        raise ValueError(f"get_association requires method to be one of {allowed_methods}")

    if 'norm' in method:
        assoc = [corrcov(a) for a in assoc]

    return [a.detach().abs() for a in assoc]
