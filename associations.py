import torch
from torch.utils.data import DataLoader
from math import ceil, prod
from tqdm import tqdm
from typing import List, Optional
import itertools


METHODS = ['forward_cov', 'forward_jac', 'backward_jac', 'backward_hess']
METHODS += [m+"_norm" for m in METHODS]


def corrcov(covariance, eps=1e-12):
    sigma = covariance.diag().sqrt()
    sigma[sigma < eps] = eps
    return covariance / (sigma.view(-1, 1) * sigma.view(1, -1))


@torch.jit.script
def sum_hessian(loss: torch.Tensor, hidden_layers: List[torch.Tensor]) -> torch.Tensor:
    """Given scalar tensor 'loss' and list of [(b,d1), (b,d2), ...] batch of hidden activations, computes (sum(d), sum(d))
     size sum of hessians, summed over the batch dimension

    We make use of the fact that grad^2(f) = grad(grad(f)) and that sum of grads = grad of sum. So, sum(grad^2(f)) is
    computed as grad(sum(grad(f))).
    """
    dims = [h.size(1) for h in hidden_layers]
    hessian = hidden_layers[0].new_zeros(sum(dims), sum(dims))
    row_offset = 0
    grad_h = torch.autograd.grad([loss], hidden_layers, create_graph=True)
    for i, h_i in enumerate(hidden_layers):
        grad_h_i = grad_h[i]
        assert grad_h_i is not None
        sum_grad = torch.sum(grad_h_i, dim=0)
        col_offset = 0
        for j, h_j in enumerate(hidden_layers):
            if j < i:
                # Lower triangle = copy of transpose of upper triangle
                hessian[row_offset:row_offset+dims[i], col_offset:col_offset+dims[j]] = \
                    hessian[col_offset:col_offset+dims[j], row_offset:row_offset+dims[i]].T
            else:
                for k in range(dims[i]):
                    hess_ijk = torch.autograd.grad([sum_grad[k]], [h_j], retain_graph=True)[0]
                    assert hess_ijk is not None
                    hessian[row_offset+k, col_offset:col_offset+dims[j]] = hess_ijk.sum(dim=0)
            col_offset += dims[j]
        row_offset += dims[i]
    return hessian


@torch.jit.script
def sum_hessian_conv(loss: torch.Tensor, conv_feature_planes: torch.Tensor, n_subsample: Optional[int] = None) -> torch.Tensor:
    """Given scalar tensor 'loss' and (b,c,h,w) batch of feature planes, computes (c,c,h*w)-size
     sum of hessians, summed over the batch dimension, separately for each x,y location.

    We make use of the fact that grad^2(f) = grad(grad(f)) and that sum of grads = grad of sum. So, sum(grad^2(f)) is
    computed as grad(sum(grad(f))), with sums taken over the batch dimension.
    """
    b, c, h, w = conv_feature_planes.size()
    hessian = conv_feature_planes.new_zeros(c, c)
    grad = torch.autograd.grad([loss], [conv_feature_planes], create_graph=True)[0]
    assert grad is not None
    sum_grad = torch.sum(grad, dim=0)
    # Short version: all_xy = [(x, y) for x in range(w) for y in range(h)] but this kind of double-iterator is not supported by
    all_yx = [divmod(i, w) for i in range(h*w)]
    if n_subsample is not None:
        assert n_subsample >= 1, "If n_subsample is not None, it must be an integer greater than or equal to 1"
        subs = torch.randperm(len(all_yx))[:n_subsample]
        all_yx = [all_yx[i] for i in subs]
    for i in range(c):
        # NOTE: this is inefficient because it computes the hessian w.r.t. all x,y,x',y' pairs of location, only for us
        # to subselect later where x==x' and y==y'. Alas, torch doesn't let us take the grad w.r.t. a subset of
        # features, since slice operations break dependency graph.
        for y, x in all_yx:
            hess_ixy = torch.autograd.grad([sum_grad[i, y, x]], [conv_feature_planes], retain_graph=True)[0]
            assert hess_ixy is not None
            hessian[i, :] = hessian[i, :] + hess_ixy[:, :, y, x].sum(dim=0)
    return hessian


@torch.jit.script
def batch_jacobian(h1: torch.Tensor, h2: torch.Tensor, preserve_shape:bool = False) -> torch.Tensor:
    """Get batch-wise jacobians of h2 with respect to h1. Both must have size (b, ?), and output will be of size
    (b, |h1|, |h2|)

    Assuming computation graph has no dependence of h2[i,...] on h1[j,...] unless i==j. This allows us to pass a lot
    of vectorization over to torch by getting grad of h2.sum() w.r.t. h1
    """
    sz1, sz2 = h1.size(), h2.size()
    if sz1[0] != sz2[0]:
        raise ValueError(f"Incompatible batch dimensions: {sz1} and {sz2}")
    b = sz1[0]
    n1, n2 = h1.numel() // b, h2.numel() // b
    h2_flat = h2.reshape(b, n2)
    jacobians = h1.new_zeros(b, n1, n2)
    for i in range(n2):
        jac_part = torch.autograd.grad([h2_flat[:, i].sum()], [h1], retain_graph=True)[0]
        assert jac_part is not None
        jacobians[:, :, i] = jac_part.reshape(b, n1)
    if preserve_shape:
        jacobians = jacobians.reshape((b,) + sz1[1:] + sz2[1:])
    return jacobians


class BatchWiseSimilarity(object):
    def __init__(self, hidden_size, norm, max_extra_dims=32):
        self.n = 0
        self.norm = norm
        if len(hidden_size) == 1:
            self.d, = hidden_size
            self.extra_dims = 1
            self.reduced_extra_dims = 1
            self.subs_x = Ellipsis
            self.conv = False
        elif len(hidden_size) == 3:
            self.d, h, w = hidden_size
            self.extra_dims = w*h
            self.reduced_extra_dims = min(max_extra_dims, w*h)
            self.subs_x = torch.randperm(self.extra_dims)[:self.reduced_extra_dims]
            self.subs_x = torch.sort(self.subs_x).values
            self.conv = True
        else:
            raise ValueError(f"Not sure how to handle hidden layer size {hidden_size}")

    def _resample_subs(self):
        self.subs_x = torch.randperm(self.extra_dims)[:self.reduced_extra_dims]
        self.subs_x = torch.sort(self.subs_x).values

    def batch_update(self, h: torch.Tensor, **kwargs) -> None:
        """Update running calculation of dxd unit associations given a bxd batch of hidden activity

        :param h: size (b,?) batch of hidden unit activity
        :return: None
        """
        raise NotImplementedError("To-be-sublcassed")

    def finalize(self) -> torch.Tensor:
        """Finalize computations after all batches have been processed, and return final d-by-d matrix of associations.

        :return: size (d,d) matrix of pairwise associations
        """
        raise NotImplementedError("To-be-sublcassed")


class Covariance(BatchWiseSimilarity):
    def __init__(self, hidden_size, norm, **kwargs):
        super().__init__(hidden_size, norm)
        # Note: if hidden_size is (c,h,w) of a conv layer, then extra_dims=h*w and we compute covariance across channels
        # separately for each location.
        self.moment1 = torch.zeros(self.d, self.reduced_extra_dims, **kwargs)
        self.moment2 = torch.zeros(self.d, self.d, self.reduced_extra_dims, **kwargs)

    def batch_update(self, batch_hidden, **kwargs):
        b = batch_hidden.size()[0]
        self.n += b
        batch_hidden = batch_hidden.detach().view(b, self.d, self.extra_dims)[:, :, self.subs_x]
        self.moment1 += batch_hidden.sum(dim=0)
        self.moment2 += torch.einsum('iax,ibx->abx', batch_hidden, batch_hidden)
        self._resample_subs()

    def finalize(self):
        norm_moment1 = self.moment1 / self.n
        norm_moment2 = self.moment2 / self.n
        moment1_outer = torch.einsum('ax,bx->abx', norm_moment1, norm_moment1)
        cov_per_location = norm_moment2 - moment1_outer
        cov = torch.mean(cov_per_location, dim=-1)
        return torch.abs(corrcov(cov) if self.norm else cov)


class UpstreamSensitivity(BatchWiseSimilarity):
    def __init__(self, hidden_size, norm, **kwargs):
        super().__init__(hidden_size, norm)
        self.n = 0
        self.inner_prod = torch.zeros(self.d, self.d, self.reduced_extra_dims, **kwargs)

    def batch_update(self, batch_hidden, *, inpt=None, **kwargs):
        b = batch_hidden.size()[0]
        self.n += b
        # Get jacobian of hidden activity w.r.t. changes in the input, then take inner product over input dims
        batch_hidden = batch_hidden.view(b, self.d, self.extra_dims)[:, :, self.subs_x]
        for i in range(self.reduced_extra_dims):
            jac_i = batch_jacobian(inpt, batch_hidden[:, :, i]).detach()
            self.inner_prod[:, :, i] += torch.einsum('...i,...j->ij', jac_i, jac_i)
        self._resample_subs()

    def finalize(self):
        sim = self.inner_prod.mean(dim=-1) / self.n
        return torch.abs(corrcov(sim) if self.norm else sim)


class DownstreamSensitivity(BatchWiseSimilarity):
    def __init__(self, hidden_size, norm, **kwargs):
        super().__init__(hidden_size, norm)
        self.n = 0
        self.inner_prod = torch.zeros(self.d, self.d, self.extra_dims, **kwargs)

    def batch_update(self, batch_hidden, *, outpt=None, **kwargs):
        b = batch_hidden.size()[0]
        self.n += b
        out_dims = outpt.numel() // b
        # Get jacobian of hidden activity w.r.t. changes in the input, then take inner product over input dims
        jac_all = batch_jacobian(batch_hidden, outpt).detach()
        jac_all = jac_all.view(b, self.d, self.extra_dims, out_dims)
        self.inner_prod += torch.einsum('bixo,bjxo->ijx', jac_all, jac_all)
        self._resample_subs()

    def finalize(self):
        sim = self.inner_prod.mean(dim=-1) / self.n
        return torch.abs(corrcov(sim) if self.norm else sim)


class LossHessian(BatchWiseSimilarity):
    def __init__(self, hidden_size, norm, **kwargs):
        super().__init__(hidden_size, norm)
        self.n = 0
        self.hess = torch.zeros(self.d, self.d, self.extra_dims, **kwargs)

    def batch_update(self, batch_hidden, *, loss=None, **kwargs):
        b = batch_hidden.size()[0]
        self.n += b
        if not self.conv:
            self.hess = self.hess + sum_hessian(loss, [batch_hidden]).unsqueeze(-1).detach()
        else:
            self.hess = self.hess + sum_hessian_conv(loss, batch_hidden).detach()
        self._resample_subs()

    def finalize(self):
        # enforce symmetry here in case of numerical imprecision
        sym_hess = (self.hess.mean(dim=-1) + self.hess.mean(dim=-1).T) / 2 / self.n
        return torch.abs(corrcov(sym_hess) if self.norm else sym_hess)



def get_similarity_by_layer(model, method, dataset, device='cpu', batch_size=200, shuffle=True):
    model.eval()
    model.to(device)
    norm = method.endswith('_norm')

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    if method in ['forward_cov', 'forward_cov_norm']:
        sim_per_layer = [Covariance(sz, norm, device=device) for sz in model.hidden_dims]
    elif method in ['backward_hess', 'backward_hess_norm']:
        sim_per_layer = [LossHessian(sz, norm, device=device) for sz in model.hidden_dims]
    elif method in ['forward_jac', 'forward_jac_norm']:
        sim_per_layer = [UpstreamSensitivity(sz, norm, device=device) for sz in model.hidden_dims]
    elif method in ['backward_jac', 'backward_jac_norm']:
        sim_per_layer = [DownstreamSensitivity(sz, norm, device=device) for sz in model.hidden_dims]
    else:
        raise ValueError(f"get_similarity_combined requires method to be one of {METHODS}")

    for im, la in tqdm(loader, total=len(loader), desc=method, leave=False):
        im.requires_grad_(True)
        im, la = im.to(device), la.to(device)
        hidden, out = model(im)
        kwargs = {'inpt': im, 'outpt': out, 'loss': model.loss_fn(im, la, out)}
        for h, sim in zip(hidden, sim_per_layer):
            sim.batch_update(h, **kwargs)

    return [sim.finalize().detach().cpu() for sim in sim_per_layer]


def get_similarity_combined(model, method, dataset, device='cpu', batch_size=200, shuffle=True):
    model.eval()
    model.to(device)

    for h in model.hidden_dims:
        if isinstance(h, tuple) and len(h) > 1:
            raise ValueError("get_similarity_combined can only handle 1-dimensional hidden layers (e.g. Linear rather than Conv2d)")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    num_batches = ceil(len(dataset)/batch_size)
    total_d = sum(map(prod, model.hidden_dims))

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
