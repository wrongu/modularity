import torch
import torch.nn.functional as F
from models import LitWrapper
from associations import get_similarity_by_layer, get_similarity_combined
from associations import METHODS as association_methods
from modularity import monte_carlo_modularity, girvan_newman, soft_num_clusters, is_valid_adjacency_matrix, \
    alignment_score, shuffled_alignment_score, sparsify, cluster_id
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from clusim.clustering import Clustering
import clusim.sim as sim
from warnings import warn
from tqdm import tqdm
from functools import lru_cache


__MOD_VERSION = 3
__ALIGN_VERSION = 6


def loss(mdl, dataset, task, device='cpu'):
    mdl.eval()
    loss = 0.0
    loader = DataLoader(dataset, pin_memory=True, batch_size=100)
    with torch.no_grad():
        for im, la in tqdm(loader, desc='Loss', total=len(dataset)//100, leave=False):
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
    loader = DataLoader(dataset, pin_memory=True, batch_size=100)
    with torch.no_grad():
        for im, la in tqdm(loader, desc='Accuracy', total=len(dataset)//100, leave=False):
            im, la = im.to(device), la.to(device)
            _, pred = mdl(im)
            ipred = torch.argsort(pred, dim=1, descending=True)[:, :topk]
            acc += torch.sum((ipred == la.view(-1,1)).float())
    return acc.item() / len(dataset)


@lru_cache(maxsize=1)
def load_model_once(checkpoint_file, data_dir, device):
    model = LitWrapper.load_from_checkpoint(checkpoint_file, map_location=device)
    # TODO - include a dataset that's just noise as a kind of reference/null
    data_train, data_val, data_test = model.get_dataset(data_dir)
    return model.to(device), data_train, data_val, data_test


def evaluate(checkpoint_file, data_dir, metrics=None, device='cpu'):
    info = torch.load(checkpoint_file)

    if metrics is None:
        metrics = ['train_loss',
                   'train_acc',
                   'val_loss',
                   'val_acc',
                   'test_loss',
                   'test_acc',
                   'l1_norm',
                   'l2_norm',
                   'sparsity',
                   'nuc_norm']

    # Loss and accuracy metrics
    if 'train_loss' not in info and 'train_loss' in metrics:
        model, data_train, _, _ = load_model_once(checkpoint_file, data_dir, device)
        info['train_loss'] = loss(model, data_train, 'sup', device=device)
    if 'train_acc' not in info and 'train_acc' in metrics:
        model, data_train, _, _ = load_model_once(checkpoint_file, data_dir, device)
        info['train_acc'] = accuracy(model, data_train, 'sup', device=device)
    if 'val_loss' not in info and 'val_loss' in metrics:
        model, _, data_val, _ = load_model_once(checkpoint_file, data_dir, device)
        info['val_loss'] = loss(model, data_val, 'sup', device=device)
    if 'val_acc' not in info and 'val_acc' in metrics:
        model, _, data_val, _ = load_model_once(checkpoint_file, data_dir, device)
        info['val_acc'] = accuracy(model, data_val, 'sup', device=device)
    if 'test_loss' not in info and 'test_loss' in metrics:
        model, _, _, data_test = load_model_once(checkpoint_file, data_dir, device)
        info['test_loss'] = loss(model, data_test, 'sup', device=device)
    if 'test_acc' not in info and 'test_acc' in metrics:
        model, _, _, data_test = load_model_once(checkpoint_file, data_dir, device)
        info['test_acc'] = accuracy(model, data_test, 'sup', device=device)

    # Weight norms, using LitWrapper.l2_norm and LitWrapper.l1_norm
    if 'l2_norm' not in info and 'l2_norm' in metrics:
        model, _, _, _ = load_model_once(checkpoint_file, data_dir, device)
        info['l2_norm'] = model.l2_norm().item()
    if 'l1_norm' not in info and 'l1_norm' in metrics:
        model, _, _, _ = load_model_once(checkpoint_file, data_dir, device)
        info['l1_norm'] = model.l1_norm().item()
    if 'sparsity' not in info and 'sparsity' in metrics:
        model, _, _, _ = load_model_once(checkpoint_file, data_dir, device)
        info['sparsity'] = model.sparsity().item()
    if 'nuc_norm' not in info and 'nuc_norm' in metrics:
        model, _, _, _ = load_model_once(checkpoint_file, data_dir, device)
        info['nuc_norm'] = model.nuc_norm().item()

    torch.save(info, checkpoint_file)
    return info


def eval_modularity(checkpoint_file, data_dir, target_entropy=None, mc_steps=5000, metrics=None, sparseness=None, align=True, combined=False, device='cpu'):
    info = torch.load(checkpoint_file)

    # Use defaults unless target_entropy is given
    mc_kwargs = {'device': device}
    if target_entropy is not None:
        mc_kwargs['target_entropy'] = target_entropy

    if sparseness is None:
        sparseness = [None]
    else:
        sparseness = [None] + list(sparseness)

    if metrics is None:
        metrics = association_methods

    # Precompute 'association' matrices and store in 'assoc' dictionary of checkpoint data
    assoc_info = info.get('assoc', {})
    for meth in metrics:
        # Compute similarity/association per layer using each requested method. Each of assoc_info[meth] will be a list
        # of similarity matrices, one per layer
        if meth not in assoc_info:
            # compute layer-wise association info
            model, _, _, data_test = load_model_once(checkpoint_file, data_dir, device)
            assoc_info[meth] = get_similarity_by_layer(model, meth, data_test, device=device, batch_size=100)
            if not all(is_valid_adjacency_matrix(m, enforce_sym=True) for m in assoc_info[meth]):
                warn(f"First sanity check on association method {meth} failed!")

        if combined:
            # ...now do it with all hidden layers stacked. Each of assoc_info[combo_key] will be a list containing a single
            # similarity matrix. We nonetheless wrap it in a list for compatibility with the per-layer methods.
            combo_key = meth + '_combined'
            if combo_key not in assoc_info:
                # compute combined association info
                model, _, _, data_test = load_model_once(checkpoint_file, data_dir, device)
                assoc_info[combo_key] = [get_similarity_combined(model, meth, data_test, device=device)]
                if not is_valid_adjacency_matrix(assoc_info[combo_key][0], enforce_sym=True):
                    warn(f"First sanity check on association method {combo_key} failed!")

    info['assoc'] = assoc_info

    # For each requested method, compute and store (1) cluster assignments and (2) modularity score
    module_info = info.get('modules', {})
    for imeth in range(len(metrics)):
        suffixes = ['', '_combined'] if combined else ['']
        for suffix in suffixes:
            meth = metrics[imeth] + suffix
            for sp in sparseness:
                key = meth if sp is None else f"{meth}.{sp:.2f}"
                if key not in module_info or module_info[key] == [] or module_info[key][0].get('version', 0) < __MOD_VERSION:
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
                                'version': __MOD_VERSION
                            })
                            continue

                        if sp is not None:
                            adj = sparsify(adj, sp)

                        print(f"Running modules.{key} on {device}")
                        try:
                            clusters, mc_scores, mc_temps, mc_ents = monte_carlo_modularity(adj, steps=mc_steps, **mc_kwargs)
                        except RuntimeError:
                            print(f"Error in modules.{key}... storing NaN values")
                            module_info[key].append({
                                'adj': adj.cpu(),
                                'sparse': sp,
                                'clusters': float('nan')*torch.ones(adj.size()),
                                'score': float('nan'),
                                'mc_scores': float('nan')*torch.ones(mc_steps),
                                'num_clusters': float('nan'),
                                'mc_temperatures': float('nan')*torch.ones(mc_steps),
                                'mc_entropies': float('nan')*torch.ones(mc_steps),
                                'version': __MOD_VERSION
                            })
                            continue


                        module_info[key].append({
                            'adj': adj.cpu(),
                            'sparse': sp,
                            'clusters': clusters.cpu(),
                            'score': girvan_newman(adj.cpu(), clusters.cpu()),
                            'mc_scores': mc_scores.cpu(),
                            'num_clusters': soft_num_clusters(clusters.cpu()),
                            'mc_temperatures': mc_temps.cpu(),
                            'mc_entropies': mc_ents.cpu(),
                            'version': __MOD_VERSION
                        })
                else:
                    print(f"Skipping modules.{key} -- already done!")
    info['modules'] = module_info

    # Compute module alignments
    if align:
        alignment_info = info.get('align', {})
        for i in range(len(metrics)):
            for j in range(i, len(metrics)):
                for suffix in ['', '_combined']:
                    meth1, meth2 = metrics[i] + suffix, metrics[j] + suffix
                    for sp in sparseness:
                        # Sort methods so key is always in alphabetical order <method a>:<method b>
                        meth_a, meth_b = min([meth1, meth2]), max([meth1, meth2])
                        key = meth_a + ":" + meth_b
                        if sp is not None:
                            key = key + f".{sp:.2f}"
                            meth_a = meth_a + f".{sp:.2f}"
                            meth_b = meth_b + f".{sp:.2f}"
                        if key not in alignment_info or alignment_info[key] == [] or alignment_info[key][0].get('version', 0) < __ALIGN_VERSION:
                            # Create or re-load a dictionary of alignment stats per layer. In the subsequent loop,
                            # writing values to 'this_align_info' modifies everything in place.
                            num_layers = len(info['modules'][meth_a])
                            alignment_info[key] = alignment_info.get(key, [{} for _ in range(num_layers)])
                            # Loop over network layers
                            for info_a, info_b, this_align_info in zip(info['modules'][meth_a], info['modules'][meth_b], alignment_info[key]):
                                adj_a, adj_b = info_a['adj'], info_b['adj']
                                c1, c2 = info_a['clusters'].to(device), info_b['clusters'].to(device)
                                if 'adj_spearman_r' not in this_align_info:
                                    triu_ij = torch.triu_indices(*adj_a.size(), offset=1)
                                    this_align_info['adj_spearman_r'], this_align_info['adj_spearman_p'] = \
                                        spearmanr(adj_a[triu_ij[0], triu_ij[1]], adj_b[triu_ij[0], triu_ij[1]])
                                if 'score' not in this_align_info:
                                    this_align_info['score'] = alignment_score(c1, c2).cpu()
                                if 'null' not in this_align_info:
                                    shuffle_scores = shuffled_alignment_score(c1, c2, n_shuffle=5000).cpu()
                                    this_align_info['null'] = shuffle_scores
                                    this_align_info['p'] = (shuffle_scores > this_align_info['score']).float().mean()
                                if 'rmi' not in this_align_info:
                                    clu_c1 = Clustering(elm2clu_dict={i:[c.item()] for i, c in enumerate(cluster_id(c1.detach()))})
                                    clu_c2 = Clustering(elm2clu_dict={i:[c.item()] for i, c in enumerate(cluster_id(c2.detach()))})
                                    this_align_info['rmi'] = sim.rmi(clu_c1, clu_c2),  # Reduced Mutual Information from clusim package
                                    this_align_info['vi'] = sim.vi(clu_c1, clu_c2),  # Variation in Information from clusim (lower=more similar)
                                    this_align_info['rmi_norm'] = sim.rmi(clu_c1, clu_c2, 'normalized'),  # Reduced Mutual Information from clusim package
                                    this_align_info['vi_norm'] = sim.vi(clu_c1, clu_c2, 'entropy'),  # Variation in Information from clusim (lower=more similar)
                                    this_align_info['element_sim'] = sim.element_sim(clu_c1, clu_c2),  # Element-centric similarity from clusim
                                if 'transfer_AaPb' not in this_align_info:
                                    this_align_info['transfer_AaPb'] = girvan_newman(info_a['adj'], info_b['clusters'])
                                    this_align_info['transfer_AbPa'] = girvan_newman(info_b['adj'], info_a['clusters'])
                                this_align_info['sparse'] = sp
                                this_align_info['version'] = __ALIGN_VERSION
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
    parser.add_argument('--metrics', default='')
    parser.add_argument('--target-entropy', default=None)
    parser.add_argument('--skip-modularity', action='store_true', default=False)
    parser.add_argument('--modularity-metrics', default='')
    parser.add_argument('--modularity-combined', action='store_true')
    parser.add_argument('--modularity-sparseness', default='')
    parser.add_argument('--skip-alignment', action='store_true', default=False)
    args = parser.parse_args()

    eval_metrics = args.metrics.split(",") if args.metrics != '' else None
    mod_metrics = args.modularity_metrics.split(",") if args.modularity_metrics != '' else None
    # Sparseness is either a list of floats, parsed from comma-separated inputs, or None
    sparseness = [float(s) for s in args.modularity_sparseness.split(",")] if args.modularity_sparseness != '' else None

    # pprint(eval_metrics)
    # pprint(mod_metrics)

    evaluate(args.ckpt_file, args.data_dir,
             metrics=eval_metrics,
             device=args.device)

    if not args.skip_modularity:
        eval_modularity(args.ckpt_file, args.data_dir,
                        metrics=mod_metrics,
                        device=args.device,
                        target_entropy=args.target_entropy,
                        sparseness=sparseness,
                        combined=args.modularity_combined,
                        align=not args.skip_alignment)

    if eval_metrics is not None:
        info = torch.load(args.ckpt_file)
        pprint({k: info[k] for k in eval_metrics if k in info})
