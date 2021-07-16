#%% Imports + config
import torch
from eval import evaluate
from pathlib import Path
from models import LitWrapper
import matplotlib.pyplot as plt
from math import sqrt
from itertools import chain


LOGS_DIR = Path('logs')
DATA_DIR = Path('data')


def last_model(model_dir):
    return model_dir / 'weights' / 'last.ckpt'


def best_model(model_dir):
    data = torch.load(last_model(model_dir))
    for v in data['callbacks'].values():
        if 'best_model_path' in v:
            return v['best_model_path']
    return None


base_config = {'dataset': 'mnist', 'task': 'sup', 'l1': 0.0, 'l2': 1e-5, 'drop': 0.0, 'runs': 9, 'x_scale': 'linear',
               'figsize': (3, 1.5)}

configs = [
    {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'train_acc', 'y_lim': [0.05, 1.0]},
    {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'val_acc', 'y_lim': [0.05, 1.0]},
    {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'test_acc', 'y_lim': [0.05, 1.0]},
    {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'train_acc', 'y_lim': [0.05, 1.0]},
    {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'val_acc', 'y_lim': [0.05, 1.0]},
    {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'test_acc', 'y_lim': [0.05, 1.0]},
    {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'l1_norm'},
    {'l2': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l2', 'y': 'l2_norm'},
    {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'l1_norm'},
    {'l1': torch.logspace(-5, -1, 9), 'x_scale': 'log', 'x': 'l1', 'y': 'l2_norm'}
]

configs = [dict(chain(base_config.items(), c.items())) for c in configs]

#%%

for conf in configs:
    plt.figure(figsize=(6, 4) if 'figsize' not in conf else conf['figsize'])
    x_name, y_name = conf['x'], conf['y']
    x_vals = conf[x_name]
    y_vals = torch.zeros(len(x_vals), conf['runs'], dtype=conf.get('dtype'))
    for r in range(conf['runs']):
        conf['run'] = r
        for i, val in enumerate(x_vals):
            conf[x_name] = val
            info = evaluate(last_model(LOGS_DIR / LitWrapper(**conf).get_uid()), data_dir=DATA_DIR, metrics=[y_name])
            if y_name in info:
                y_vals[i, r] = info[y_name]
            elif y_name in info['hyper_parameters']:
                y_vals[i, r] = info['hyper_parameters'][y_name]
    y_vals = y_vals.detach().cpu()
    plt.plot(x_vals, y_vals, linewidth=0.5)
    try:
        mu_y, sem_y = y_vals.mean(dim=1), y_vals.std(dim=1) / sqrt(y_vals.size(1))
        plt.errorbar(x_vals, mu_y, yerr=sem_y, fmt='-k', marker='.', linewidth=1.5)
    except RuntimeError:
        print(f"Cannot do errorbars on {y_name}")
    plt.xscale(conf['x_scale'])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    if 'y_lim' in conf:
        plt.ylim(conf['y_lim'])
    plt.title(f"{conf['y']} vs {conf['x']} for {conf['dataset']} {conf['task']}")
    plt.savefig(f"figures/{conf['dataset']}_{conf['task']}__{conf['y']}__vs__{conf['x']}.svg")
    plt.show()
