import torch
from torch.utils.data import DataLoader
from hessian import hessian
from math import ceil
from tqdm import tqdm


# TODO - consider refactoring to use pl.Trainer.test() with callbacks


def corrcov(covariance, eps=1e-12):
    sigma = covariance.diag().sqrt()
    sigma[sigma < eps] = eps
    return covariance / (sigma.view(-1, 1) * sigma.view(1, -1))


# TODO - include covariance in feature space (HSIC) for both fwd and bwd methods
# TODO - include 'online' methods
def get_associations(model, method, dataset, device='cpu', batch_size=200):
    model.eval()
    model.to(device)
    allowed_methods = ['forward_cov', 'forward_cov_norm', 'backward_hess', 'backward_hess_norm']

    if method not in allowed_methods:
        raise ValueError(f"get_association requires method to be one of {allowed_methods}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    num_batches = ceil(len(dataset)/batch_size)

    # Run a single test input to get hidden dimension for each of the model.HIDDEN_LAYERS
    h_depths = model.HIDDEN_LAYERS
    tmp_im, _ = dataset[0]
    tmp_h = model(tmp_im.unsqueeze(0).to(device), depth=h_depths)
    h_dims = [h.numel() for h in tmp_h]

    if method in ['forward_cov', 'forward_cov_norm']:
        moment1 = [torch.zeros(d, device=device) for d in h_dims]
        moment2 = [torch.zeros(d, d, device=device) for d in h_dims]
        with torch.no_grad():
            for im, _ in tqdm(loader, desc=method, total=num_batches):
                hidden, _ = model(im.to(device), depth=h_depths)

                for h, m1, m2 in zip(hidden, moment1, moment2):
                    m1 += h.sum(dim=0) / len(dataset)
                    m2 += h.T @ h / len(dataset)

            # Convert from moment1 and moment2 to covariance for each hidden layer
            assoc = [m2 - m1.view(-1, 1) * m1.view(1, -1) for m1, m2 in zip(moment1, moment2)]

    elif method in ['backward_hess', 'backward_hess_norm']:
        assoc = [torch.zeros(d, d, device=device) for d in h_dims]
        with torch.no_grad():
            for im, la in tqdm(loader, desc=method, total=num_batches):
                im, la = im.to(device), la.to(device)
                hidden, out = model(im, depth=h_depths)
                loss = model.loss_fn(im, la, out)
                for hid, hess in zip(hidden, assoc):
                    hess += hessian(loss, hid) / len(dataset)

    if 'norm' in method:
        assoc = [corrcov(a) for a in assoc]

    return [a.abs() for a in assoc()]
