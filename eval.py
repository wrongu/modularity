import torch
import torch.nn.functional as F
from models import LitWrapper
from associations import get_associations
from modularity import monte_carlo_modularity, girvan_newman, soft_num_clusters, is_valid_adjacency_matrix
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
        info['l2_norm'] = model.l2_norm().detach()
    if 'l1_norm' not in info and (metrics is None or 'l1_norm' in metrics):
        info['l1_norm'] = model.l1_norm().detach()

    torch.save(info, checkpoint_file)
    return info


def eval_modularity(checkpoint_file, data_dir, metrics=None, device='cpu'):
    info = torch.load(checkpoint_file)
    model = LitWrapper.load_from_checkpoint(checkpoint_file)
    _, _, data_test = model.get_dataset(data_dir)

    assoc_methods = ['forward_cov', 'forward_cov_norm', 'backward_hess', 'backward_hess_norm']
    assoc_info = info.get('assoc', {})

    # Precompute 'association' matrices and store in 'assoc' dictionary of checkpoint data
    for meth in assoc_methods:
        if meth not in assoc_info and (metrics is None or meth in metrics):
            assoc_info[meth] = get_associations(model, meth, data_test, device=device)
            if not all(is_valid_adjacency_matrix(m, enforce_sym=True) for m in assoc_info[meth]):
                raise RuntimeError(f"Sanity check on association method {meth} failed!")
    info['assoc'] = assoc_info

    # For each requested method, compute and store (1) cluster assignments and (2) modularity score
    module_info = info.get('modules', {})
    for meth in assoc_methods:
        if meth not in module_info and (metrics is None or meth in metrics):
            module_info[meth] = []
            for adj in info['assoc'][meth]:
                adj = adj - adj.diag().diag()
                if not is_valid_adjacency_matrix(adj, enforce_sym=True, enforce_no_self=True):
                    raise RuntimeError(f"Sanity check on association method {meth} failed!")
                clusters, mc_scores = monte_carlo_modularity(adj, steps=100000, temperature=2e-4)
                module_info[meth].append({
                    'adj': adj.cpu(),
                    'clusters': clusters.cpu(),
                    'score': girvan_newman(adj, clusters).cpu(),
                    'mc_scores': mc_scores.cpu(),
                    'num_clusters': soft_num_clusters(clusters).cpu(),
                    'temperature': 2e-4
                })
    info['modules'] = module_info

    torch.save(info, checkpoint_file)
    return info


if __name__ == '__main__':
    from pprint import pprint
    from pathlib import Path
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--ckpt-file', metavar='CKPT', type=Path, required=True)
    parser.add_argument('--data-dir', default=Path('data'), metavar='DATA', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--metrics', default='train_acc,val_acc,test_acc,l1_norm,l2_norm')
    parser.add_argument('--modularity-metrics', default='')
    args = parser.parse_args()

    eval_metrics = args.metrics.split(",") if args.metrics != '' else []
    mod_metrics = args.modularity_metrics.split(",") if args.modularity_metrics != '' else []

    pprint(eval_metrics)
    pprint(mod_metrics)

    evaluate(args.ckpt_file, args.data_dir, eval_metrics)
    eval_modularity(args.ckpt_file, args.data_dir, mod_metrics, device=args.device)
    info = torch.load(args.ckpt_file)
    pprint({k: info[k] for k in eval_metrics if k in info})
