import torch
import torch.nn.functional as F
from models import LitWrapper
from associations import get_associations
from modularity import monte_carlo_modularity, girvan_newman, is_valid_adjacency_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm


def loss(mdl, dataset, task, device='cpu'):
    mdl.eval()
    loss = 0.0
    loader = DataLoader(dataset, batch_size=500)
    for im, la in tqdm(loader, desc='Loss', total=len(dataset)//500, leave=False):
        im, la = im.to(device), la.to(device)
        _, out = mdl(im)
        if task[:3] == 'sup':
            loss += F.cross_entropy(out, la, reduction='sum')
        elif task[:5] == 'unsup':
            loss += F.mse_loss(out, im.view(im.size(0), -1), reduction='sum')
    return loss.item() / len(dataset)


def accuracy(mdl, dataset, task, topk=1, device='cpu'):
    mdl.eval()
    if task[:5] == 'unsup':
        return torch.tensor(float('nan'))
    acc = 0.0
    loader = DataLoader(dataset, batch_size=500)
    for im, la in tqdm(loader, desc='Accuracy', total=len(dataset)//500, leave=False):
        im, la = im.to(device), la.to(device)
        _, pred = mdl(im)
        ipred = torch.argsort(pred, dim=1, descending=True)[:, :topk]
        acc += torch.sum((ipred == la.view(-1,1)).float())
    return acc.item() / len(dataset)


def evaluate(checkpoint_file, data_dir, metrics=None):
    info = torch.load(checkpoint_file)
    model = LitWrapper.load_from_checkpoint(checkpoint_file)
    data_train, data_val, data_test = model.get_dataset(data_dir)

    # Loss and accuracy metrics
    if 'train_loss' not in info and (metrics is None or 'train_loss' in metrics):
        info['train_loss'] = loss(model, data_train, model.hparams.task)
    if 'train_acc' not in info and (metrics is None or 'train_acc' in metrics):
        info['train_acc'] = accuracy(model, data_train, model.hparams.task)
    if 'val_loss' not in info and (metrics is None or 'val_loss' in metrics):
        info['val_loss'] = loss(model, data_val, model.hparams.task)
    if 'val_acc' not in info and (metrics is None or 'val_acc' in metrics):
        info['val_acc'] = accuracy(model, data_val, model.hparams.task)
    if 'test_loss' not in info and (metrics is None or 'test_loss' in metrics):
        info['test_loss'] = loss(model, data_test, model.hparams.task)
    if 'test_acc' not in info and (metrics is None or 'test_acc' in metrics):
        info['test_acc'] = accuracy(model, data_test, model.hparams.task)
    if 'test_acc' not in info and (metrics is None or 'test_acc' in metrics):
        info['test_acc'] = accuracy(model, data_test, model.hparams.task)

    # Weight norms, using LitWrapper.l2_norm and LitWrapper.l1_norm
    if 'l2_norm' not in info and (metrics is None or 'l2_norm' in metrics):
        info['l2_norm'] = model.l2_norm()
    if 'l1_norm' not in info and (metrics is None or 'l1_norm' in metrics):
        info['l1_norm'] = model.l1_norm()

    torch.save(info, checkpoint_file)
    return info


def eval_modularity(checkpoint_file, data_dir, metrics=None):
    info = torch.load(checkpoint_file)
    model = LitWrapper.load_from_checkpoint(checkpoint_file)
    data_train, data_val, data_test = model.get_dataset(data_dir)

    assoc_methods = ['forward_cov', 'forward_cov_norm', 'backward_cov', 'backward_cov_norm']
    assoc_info = info.get('assoc', {})

    # Precompute 'association' matrices and store in 'assoc' dictionary of checkpoint data
    for meth in assoc_methods:
        if meth not in assoc_info and (metrics is None or meth in metrics):
            assoc_info[meth] = get_associations(model, meth, data_test)
            if not is_valid_adjacency_matrix(assoc_info[meth], enforce_sym=True):
                raise RuntimeError(f"Sanity check on association method {meth} failed!")
    info['assoc'] = assoc_info

    # For each requested method, compute and store (1) cluster assignments and (2) modularity score
    module_info = info.get('modules', {})
    for meth in assoc_methods:
        if meth not in module_info and (metrics is None or meth in metrics):
            adj = assoc_info[meth] - assoc_info[meth].diag()
            if not is_valid_adjacency_matrix(adj, enforce_sym=True, enforce_no_self=True):
                raise RuntimeError(f"Sanity check on association method {meth} failed!")
            clusters, mc_scores = monte_carlo_modularity(adj, steps=50000, temperature=1e-4)
            module_info[meth] = {
                'adj': adj,
                'clusters': clusters,
                'score': girvan_newman(adj, clusters),
                'mc_scores': mc_scores,
                'temperature': 1e-4
            }
    info['modules'] = module_info

    torch.save(info, checkpoint_file)
    return info


if __name__ == '__main__':
    from pprint import pprint
    from pathlib import Path
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--ckpt-file', metavar='CKPT', type=Path)
    parser.add_argument('--data-dir', metavar='DATA', type=Path)
    parser.add_argument('--metrics', default='train_acc,val_acc,test_acc,l1_norm,l2_norm')
    parser.add_argument('--modularity-metrics', default='')
    args = parser.parse_args()

    eval_metrics = [s.split() for s in args.metrics.split(",")] if args.metrics != '' else []
    mod_metrics = [s.split() for s in args.modularity_metrics.split(",")] if args.modularity_metrics != '' else []

    evaluate(args.ckpt_file, args.data_dir, eval_metrics)
    eval_modularity(args.ckpt_file, args.data_dir, mod_metrics)
    info = torch.load(args.ckpt_file)
    pprint({k: info[k] for k in eval_metrics if k in info})
