#%% Imports + config
import torch
from eval import evaluate, eval_modularity
from pathlib import Path
from models import LitWrapper
import matplotlib.pyplot as plt
from itertools import chain


LOGS_DIR = Path('logs')
DATA_DIR = Path('data')
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'


def last_model(model_dir):
    return model_dir / 'weights' / 'last.ckpt'


def best_model(model_dir):
    data = torch.load(last_model(model_dir))
    for v in data['callbacks'].values():
        if 'best_model_path' in v:
            return v['best_model_path']
    return None


def get_metric(ckpt_file, name, call_eval=True):
    nested_names = name.split('.')
    if not call_eval:
        # Just load the file as-is, hoping that stuff has already been computed
        info = torch.load(ckpt_file)
    elif nested_names[0] == "modules":
        # Stuff inside info["modules"] comes from
        assert len(nested_names) == 4, "name must be modules.[assoc_method].[layer_number].[field_name]"
        info = eval_modularity(ckpt_file, data_dir=DATA_DIR, metrics=nested_names[1:2], device=DEVICE)
    else:
        info = evaluate(ckpt_file, data_dir=DATA_DIR, metrics=nested_names[0:1])

    if nested_names[0] == "modules" and "modules" in info and nested_names[1] in info["modules"]:
        return info["modules"][nested_names[1]][int(nested_names[2])][nested_names[3]]
    elif name in info["hyper_parameters"]:
        return info["hyper_parameters"][name]
    elif name in info:
        return info[name]
    else:
        return float('nan')


def nanstats(data):
    valid = ~torch.isnan(data)
    count = torch.sum(valid.float(), dim=-1)
    mean = torch.nansum(data, dim=-1) / count
    std = torch.zeros(mean.size())
    for i in range(len(std)):
        std[i] = torch.sqrt(torch.sum((data[i, valid[i, :]] - mean[i])**2) / (count[i] - 1))
    sem = std / torch.sqrt(count)
    return mean, std, sem


base_config = {'dataset': 'mnist', 'task': 'sup', 'l1': 0.0, 'l2': 1e-5, 'drop': 0.0, 'runs': 9, 'x_scale': 'linear',
               'figsize': (3, 1.5), 'eval': False}

# configs = [
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'train_acc', 'y_lim': [0.05, 1.0]},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'val_acc', 'y_lim': [0.05, 1.0]},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'test_acc', 'y_lim': [0.05, 1.0]},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'train_acc', 'y_lim': [0.05, 1.0]},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'val_acc', 'y_lim': [0.05, 1.0]},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'test_acc', 'y_lim': [0.05, 1.0]},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'l1_norm'},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'l2_norm'},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'l1_norm'},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'l2_norm'},
# ]

configs = [
    {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.forward_cov.0.score', 'y_lim': [0.05, 0.25]},
    {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.forward_cov.0.score', 'y_lim': [0.05, 0.25]},
    {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.forward_cov_norm.0.score', 'y_lim': [0.05, 0.25]},
    {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.forward_cov_norm.0.score', 'y_lim': [0.05, 0.25]},
    {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.backward_hess.0.score', 'y_lim': [0.05, 0.25]},
    {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.backward_hess.0.score', 'y_lim': [0.05, 0.25]},
    {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.backward_hess_norm.0.score', 'y_lim': [0.05, 0.25]},
    {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.backward_hess_norm.0.score', 'y_lim': [0.05, 0.25]},
    {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.forward_cov.1.score', 'y_lim': [0.05, 0.25]},
    {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.forward_cov.1.score', 'y_lim': [0.05, 0.25]},
    {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.forward_cov_norm.1.score', 'y_lim': [0.05, 0.25]},
    {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.forward_cov_norm.1.score', 'y_lim': [0.05, 0.25]},
    {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.backward_hess.1.score', 'y_lim': [0.05, 0.25]},
    {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.backward_hess.1.score', 'y_lim': [0.05, 0.25]},
    {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.backward_hess_norm.1.score', 'y_lim': [0.05, 0.25]},
    {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.backward_hess_norm.1.score', 'y_lim': [0.05, 0.25]}
]

# configs = [
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.forward_cov.0.num_clusters', 'y_lim': [0, 20]},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.forward_cov.0.num_clusters', 'y_lim': [0, 20]},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.forward_cov_norm.0.num_clusters', 'y_lim': [0, 20]},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.forward_cov_norm.0.num_clusters', 'y_lim': [0, 20]},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.backward_hess.0.num_clusters', 'y_lim': [0, 20]},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.backward_hess.0.num_clusters', 'y_lim': [0, 20]},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.backward_hess_norm.0.num_clusters', 'y_lim': [0, 20]},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.backward_hess_norm.0.num_clusters', 'y_lim': [0, 20]},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.forward_cov.1.num_clusters', 'y_lim': [0, 20]},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.forward_cov.1.num_clusters', 'y_lim': [0, 20]},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.forward_cov_norm.1.num_clusters', 'y_lim': [0, 20]},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.forward_cov_norm.1.num_clusters', 'y_lim': [0, 20]},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.backward_hess.1.num_clusters', 'y_lim': [0, 20]},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.backward_hess.1.num_clusters', 'y_lim': [0, 20]},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.backward_hess_norm.1.num_clusters', 'y_lim': [0, 20]},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.backward_hess_norm.1.num_clusters', 'y_lim': [0, 20]},
# ]

# configs = [
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.forward_cov.0.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.forward_cov.0.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.forward_cov_norm.0.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.forward_cov_norm.0.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.backward_hess.0.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.backward_hess.0.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.backward_hess_norm.0.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.backward_hess_norm.0.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.forward_cov.1.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.forward_cov.1.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.forward_cov_norm.1.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.forward_cov_norm.1.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.backward_hess.1.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.backward_hess.1.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'modules.backward_hess_norm.1.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
#     {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'modules.backward_hess_norm.1.temperature', 'y_lim': [1e-4, 1e-2], 'y_scale': 'log'},
# ]

configs = [dict(chain(base_config.items(), c.items())) for c in configs]

#%%

torch.set_grad_enabled(False)
for conf in configs:
    plt.figure(figsize=(6, 4) if 'figsize' not in conf else conf['figsize'])
    x_name, y_name = conf['x'], conf['y']
    x_vals = conf[x_name]
    y_vals = torch.zeros(len(x_vals), conf['runs'], dtype=conf.get('dtype'))
    for r in range(conf['runs']):
        conf['run'] = r
        for i, val in enumerate(x_vals):
            conf[x_name] = val
            y_vals[i, r] = get_metric(last_model(LOGS_DIR / LitWrapper(**conf).get_uid()), name=y_name, call_eval=conf['eval'])
    y_vals = y_vals.detach().cpu()
    plt.plot(x_vals, y_vals, linewidth=0.5)
    try:
        mu_y, _, sem_y = nanstats(y_vals)
        plt.errorbar(x_vals, mu_y, yerr=sem_y, fmt='-k', marker='.', linewidth=1.5)
    except RuntimeError:
        print(f"Cannot do errorbars on {y_name}")
    plt.xscale(conf['x_scale'])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    if 'y_lim' in conf:
        plt.ylim(conf['y_lim'])
    if 'y_scale' in conf:
        plt.yscale(conf['y_scale'])
    plt.title(f"{conf['y']} vs {conf['x']} for {conf['dataset']} {conf['task']}")
    plt.savefig(f"figures/{conf['dataset']}_{conf['task']}__{conf['y']}__vs__{conf['x']}.svg")
    plt.show()
