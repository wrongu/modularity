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


def batch_jacobian(h1, h2):
    """Get batch-wise jacobians of h2 with respect to h1. Both must have size (b, ?), and output will be of size
    (b, |h1|, |h2|)

    Assuming computation graph has no dependence of h2[i,...] on h1[j,...] unless i==j. This allows us to pass a lot
    of vectorization over to torch by getting grad of h2.sum() w.r.t. h1
    """
    b = h1.size(0)
    h2 = h2.reshape(b, -1)
    d1, d2 = h1.numel()//b, h2.size(1)
    jacobians = h1.new_zeros(b, d1, d2)
    for i in range(d2):
        jac_part = torch.autograd.grad(h2[:, i].sum(), h1, retain_graph=True)[0]
        jacobians[:, :, i] = jac_part.reshape(b, d1)
    return jacobians


# TODO - include covariance in feature space (HSIC) for both fwd and bwd methods
# TODO - include 'online' methods
def get_associations(model, method, dataset, device='cpu', batch_size=200, shuffle=True):
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
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
                a += sum_hessian(loss, h) / len(dataset)
        # Ensure symmetry, since hessians will not be *exactly* symmetric up to floating point errors, and we will
        # be asserting symmetry later.
        assoc = [(a + a.T)/2. for a in assoc]
    elif method in ['forward_jac', 'forward_jac_norm']:
        assoc = [torch.zeros(d, d, device=device) for d in model.hidden_dims]
        for im, la in tqdm(loader, desc=method, total=num_batches):
            im, la = im.to(device), la.to(device)
            im.requires_grad_(True)
            hidden, out = model(im)
            for a, h in zip(assoc, hidden):
                jac = batch_jacobian(im, h)
                a += torch.einsum('bix,biy->xy', jac, jac) / len(dataset)
    elif method in ['backward_jac', 'backward_jac_norm']:
        assoc = [torch.zeros(d, d, device=device) for d in model.hidden_dims]
        for im, la in tqdm(loader, desc=method, total=num_batches):
            im, la = im.to(device), la.to(device)
            hidden, out = model(im)
            for a, h in zip(assoc, hidden):
                jac = batch_jacobian(h, out)
                a += torch.einsum('bxi,byi->xy', jac, jac) / len(dataset)
    else:
        allowed_methods = ['forward_cov', 'forward_cov_norm', 'backward_hess', 'backward_hess_norm',
                           'forward_jac', 'forward_jac_norm', 'backward_jac', 'backward_jac_norm']
        raise ValueError(f"get_association requires method to be one of {allowed_methods}")

    if 'norm' in method:
        assoc = [corrcov(a) for a in assoc]

    return [a.detach().abs() for a in assoc]
