import torch
from torch.utils.data import DataLoader
from math import ceil
from tqdm import tqdm
from typing import List
import itertools


def corrcov(covariance, eps=1e-12):
    sigma = covariance.diag().sqrt()
    sigma[sigma < eps] = eps
    return covariance / (sigma.view(-1, 1) * sigma.view(1, -1))


def sum_hessian(loss, hidden_layers: List[torch.Tensor]):
    """Given scalar tensor 'loss' and list of [(b,d1), (b,d2), ...] batch of hidden activations, computes (sum(d), sum(d))
     size sum of hessians, summed over the batch dimension

    We make use of the fact that grad^2(f) = grad(grad(f)) and that sum of grads = grad of sum. So, sum(grad^2(f)) is
    computed as grad(sum(grad(f))).
    """
    dims = [h.size(1) for h in hidden_layers]
    hessian = hidden_layers[0].new_zeros(sum(dims), sum(dims))
    row_offset = 0
    for i, h_i in enumerate(hidden_layers):
        grad_i = torch.autograd.grad(loss, h_i, retain_graph=True, create_graph=True)[0]
        sum_grad = torch.sum(grad_i, dim=0)
        col_offset = 0
        for j, h_j in enumerate(hidden_layers):
            if j < i:
                # Lower triangle = copy of transpose of upper triangle
                hessian[row_offset:row_offset+dims[i], col_offset:col_offset+dims[j]] = \
                    hessian[col_offset:col_offset+dims[j], row_offset:row_offset+dims[i]].T
            else:
                for k in range(dims[i]):
                    hessian[row_offset+k, col_offset:col_offset+dims[j]] = \
                        torch.autograd.grad(sum_grad[k], h_j, retain_graph=True)[0].sum(dim=0)
            col_offset += dims[j]
        row_offset += dims[i]
    return hessian


def sum_hessian_conv_avg_over_space(loss, conv_feature_planes: torch.Tensor):
    """Given scalar tensor 'loss' and (b,c,h,w) batch of feature planes, computes (c,c)-size
     sum of hessians, summed over the batch dimension and averaged over spatial dimensions.

    We make use of the fact that grad^2(f) = grad(grad(f)) and that sum of grads = grad of sum. So, sum(grad^2(f)) is
    computed as grad(sum(grad(f))), with sums taken over the batch dimension.
    """
    b, c, h, w = conv_feature_planes.size()
    hessian = conv_feature_planes.new_zeros(c, c)
    grad = torch.autograd.grad(loss, conv_feature_planes, retain_graph=True, create_graph=True)[0]
    sum_grad = torch.sum(grad, dim=0)
    for i in range(c):
        # NOTE: this is inefficient because it computes the hessian w.r.t. all x,y,x',y' pairs of location, only for us
        # to subselect later where x==x' and y==y'. Alas, torch doesn't let us take the grad w.r.t. a subset of
        # features, since slice operations break dependency graph.
        for x, y in tqdm(itertools.product(range(w), range(h)), total=h*w, desc=f'xy[{i}]', leave=False):
            hess_ixy = torch.autograd.grad(sum_grad[i, y, x], conv_feature_planes, retain_graph=True)[0]
            hessian[i, :] = hessian[i, :] + hess_ixy[:, :, y, x].sum(dim=0)
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


def get_similarity_by_layer(model, method, dataset, device='cpu', batch_size=200, shuffle=True):
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
        raise ValueError(f"get_similarity_combined requires method to be one of {allowed_methods}")

    if 'norm' in method:
        assoc = [corrcov(a) for a in assoc]

    return [a.detach().abs() for a in assoc]


def get_similarity_combined(model, method, dataset, device='cpu', batch_size=200, shuffle=True):
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    num_batches = ceil(len(dataset)/batch_size)
    total_d = sum(model.hidden_dims)

    if method in ['forward_cov', 'forward_cov_norm']:
        moment1 = torch.zeros(total_d, device=device)
        moment2 = torch.zeros(total_d, total_d, device=device)
        for im, _ in tqdm(loader, desc=method+"_combined", total=num_batches):
            hidden, _ = model(im.to(device))
            hidden = torch.cat(hidden, dim=-1).detach()
            moment1 += hidden.sum(dim=0) / len(dataset)
            moment2 += hidden.T @ hidden / len(dataset)

        # Convert from moment1 and moment2 to covariance
        assoc = moment2 - moment1.view(-1, 1) * moment1.view(1, -1)

    elif method in ['backward_hess', 'backward_hess_norm']:
        assoc = torch.zeros(total_d, total_d, device=device)
        for im, la in tqdm(loader, desc=method+"_combined", total=num_batches):
            im, la = im.to(device), la.to(device)
            hidden, out = model(im)
            loss = model.loss_fn(im, la, out)
            assoc += sum_hessian(loss, hidden) / len(dataset)
        # Ensure symmetry, since hessians will not be *exactly* symmetric up to floating point errors, and we will
        # be asserting symmetry later.
        assoc = (assoc + assoc.T) / 2

    elif method in ['forward_jac', 'forward_jac_norm']:
        assoc = torch.zeros(total_d, total_d, device=device)
        for im, la in tqdm(loader, desc=method+"_combined", total=num_batches):
            im, la = im.to(device), la.to(device)
            im.requires_grad_(True)
            hidden, out = model(im)
            # Each jacobian is size (batch, input_dim, hidden_dim). Concat along dimension [2].
            jac = torch.cat([batch_jacobian(im, h) for h in hidden], dim=2)
            assoc += torch.einsum('bix,biy->xy', jac, jac) / len(dataset)
    elif method in ['backward_jac', 'backward_jac_norm']:
        assoc = torch.zeros(total_d, total_d, device=device)
        for im, la in tqdm(loader, desc=method+"_combined", total=num_batches):
            im, la = im.to(device), la.to(device)
            hidden, out = model(im)
            # Each jacobian is size (batch, hidden_dim, output_dim). Concat along dimension [1].
            jac = torch.cat([batch_jacobian(h, out) for h in hidden], dim=1)
            assoc += torch.einsum('bxi,byi->xy', jac, jac) / len(dataset)
    else:
        allowed_methods = ['forward_cov', 'forward_cov_norm', 'backward_hess', 'backward_hess_norm',
                           'forward_jac', 'forward_jac_norm', 'backward_jac', 'backward_jac_norm']
        raise ValueError(f"get_similarity_combined requires method to be one of {allowed_methods}")

    if 'norm' in method:
        assoc = corrcov(assoc)

    return assoc.detach().abs()
