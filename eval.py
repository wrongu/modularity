import torch
import torch.nn.functional as F
from models import LitWrapper
from associations import get_associations
from modularity import monte_carlo_modularity, girvan_newman, soft_num_clusters, is_valid_adjacency_matrix, \
    alignment_score, shuffled_alignment_score, sparsify
from torch.utils.data import DataLoader
from warnings import warn
from tqdm import tqdm


__VERSION = 3


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

    if metrics is None:
        metrics = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'l1_norm', 'l2_norm']

    # Loss and accuracy metrics
    if 'train_loss' not in info and 'train_loss' in metrics:
        info['train_loss'] = loss(model, data_train, model.hparams.task)
    if 'train_acc' not in info and 'train_acc' in metrics:
        info['train_acc'] = accuracy(model, data_train, model.hparams.task)
    if 'val_loss' not in info and 'val_loss' in metrics:
        info['val_loss'] = loss(model, data_val, model.hparams.task)
    if 'val_acc' not in info and 'val_acc' in metrics:
        info['val_acc'] = accuracy(model, data_val, model.hparams.task)
    if 'test_loss' not in info and 'test_loss' in metrics:
        info['test_loss'] = loss(model, data_test, model.hparams.task)
    if 'test_acc' not in info and 'test_acc' in metrics:
        info['test_acc'] = accuracy(model, data_test, model.hparams.task)
    if 'test_acc' not in info and 'test_acc' in metrics:
        info['test_acc'] = accuracy(model, data_test, model.hparams.task)

    # Weight norms, using LitWrapper.l2_norm and LitWrapper.l1_norm
    if 'l2_norm' not in info and 'l2_norm' in metrics:
        info['l2_norm'] = model.l2_norm().detach()
    if 'l1_norm' not in info and 'l1_norm' in metrics:
        info['l1_norm'] = model.l1_norm().detach()

    torch.save(info, checkpoint_file)
    return info


def eval_modularity(checkpoint_file, data_dir, target_entropy=None, mc_steps=5000, metrics=None, sparseness=None, align=True, device='cpu'):
    info = torch.load(checkpoint_file)
    model = LitWrapper.load_from_checkpoint(checkpoint_file)
    _, _, data_test = model.get_dataset(data_dir)

    # Use defaults unless target_entropy is given
    mc_kwargs = {'device': device}
    if target_entropy is not None:
        mc_kwargs['target_entropy'] = target_entropy

    if sparseness is None:
        sparseness = [None]
    else:
        sparseness = [None] + list(sparseness)

    if metrics is None:
        metrics = ['forward_cov', 'forward_cov_norm', 'backward_hess', 'backward_hess_norm',
                   'forward_jac', 'forward_jac_norm', 'backward_jac', 'backward_jac_norm']

    # Precompute 'association' matrices and store in 'assoc' dictionary of checkpoint data
    assoc_info = info.get('assoc', {})
    for meth in metrics:
        if meth not in assoc_info:
            assoc_info[meth] = get_associations(model, meth, data_test, device=device)
            if not all(is_valid_adjacency_matrix(m, enforce_sym=True) for m in assoc_info[meth]):
                warn(f"First sanity check on association method {meth} failed!")
    info['assoc'] = assoc_info

    # For each requested method, compute and store (1) cluster assignments and (2) modularity score
    module_info = info.get('modules', {})
    for meth in metrics:
        for sp in sparseness:
            key = meth if sp is None else f"{meth}.{sp:.2f}"
            if key not in module_info or module_info[key] == [] or module_info[key][0].get('version', 0) < __VERSION:
                module_info[key] = []
                for adj in info['assoc'][meth]:
                    adj = adj - adj.diag().diag()
                    if not is_valid_adjacency_matrix(adj, enforce_sym=True, enforce_no_self=True):
                        warn(f"Second sanity check on association method {meth} failed!")
                        module_info[key].append({
                            'adj': adj.cpu(),
                            'sparse': sp,
                            'clusters': float('nan')*torch.ones(adj.size()),
                            'score': float('nan'),
                            'mc_scores': float('nan')*torch.ones(mc_steps),
                            'num_clusters': float('nan'),
                            'mc_temperatures': float('nan')*torch.ones(mc_steps),
                            'mc_entropies': float('nan')*torch.ones(mc_steps),
                            'version': __VERSION
                        })
                        continue

                    if sp is not None:
                        adj = sparsify(adj, sp)

                    print(f"Running modules.{key} on {device}")
                    clusters, mc_scores, mc_temps, mc_ents = monte_carlo_modularity(adj, steps=mc_steps, **mc_kwargs)

                    module_info[key].append({
                        'adj': adj.cpu(),
                        'sparse': sp,
                        'clusters': clusters.cpu(),
                        'score': girvan_newman(adj.cpu(), clusters.cpu()),
                        'mc_scores': mc_scores.cpu(),
                        'num_clusters': soft_num_clusters(clusters.cpu()),
                        'mc_temperatures': mc_temps.cpu(),
                        'mc_entropies': mc_ents.cpu(),
                        'version': __VERSION
                    })
            else:
                print(f"Skipping modules.{key} -- already done!")
    info['modules'] = module_info

    # Compute module alignments
    if align:
        alignment_info = info.get('align', {})
        for i, meth1 in enumerate(metrics):
            for meth2 in metrics[i:]:
                # Sort methods so key is always in alphabetical order <method a>:<method b>
                meth_a, meth_b = min([meth1, meth2]), max([meth1, meth2])
                key = meth_a + ":" + meth_b
                if key not in alignment_info or alignment_info[key] == [] or alignment_info[key][0].get('version', 0) < __VERSION:
                    alignment_info[key] = []
                    # Loop over network layers
                    for info1, info2 in zip(info['modules'][meth_a], info['modules'][meth_b]):
                        c1, c2 = info1['clusters'].to(device), info2['clusters'].to(device)
                        score = alignment_score(c1, c2).cpu()
                        shuffle_scores = shuffled_alignment_score(c1, c2, n_shuffle=5000).cpu()
                        alignment_info[key].append({
                            'score': score,
                            'p': (shuffle_scores > score).float().mean(),
                            'null': shuffle_scores,
                            'version': __VERSION
                        })
        info['align'] = alignment_info

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
    parser.add_argument('--target-entropy', default=None)
    parser.add_argument('--modularity-metrics', default='')
    parser.add_argument('--modularity-sparseness', default='')
    parser.add_argument('--skip-alignment', action='store_true', default=False)
    args = parser.parse_args()

    eval_metrics = args.metrics.split(",") if args.metrics != '' else []
    mod_metrics = args.modularity_metrics.split(",") if args.modularity_metrics != '' else []
    # Sparseness is either a list of floats, parsed from comma-separated inputs, or None
    sparseness = [float(s) for s in args.modularity_sparseness.split(",")] if args.modularity_sparseness != '' else None

    pprint(eval_metrics)
    pprint(mod_metrics)

    evaluate(args.ckpt_file, args.data_dir, eval_metrics)
    eval_modularity(args.ckpt_file, args.data_dir,
                    metrics=mod_metrics,
                    device=args.device,
                    target_entropy=args.target_entropy,
                    sparseness=sparseness,
                    align=not args.skip_alignment)
    info = torch.load(args.ckpt_file)
    pprint({k: info[k] for k in eval_metrics if k in info})
