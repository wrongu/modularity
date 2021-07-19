import torch
import numpy as np
import pandas as pd
from analysis import generate_model_specs, load_data_as_table
from pathlib import Path
import matplotlib.pyplot as plt


LOGS_DIR = Path('logs')
DATA_DIR = Path('data')
FIG_SIZE = (6, 4)


def plot_by_hyper(df: pd.DataFrame, x_name, y_name, **kwargs):
    fig = plt.figure(figsize=kwargs.get('figsize', FIG_SIZE))
    ax = fig.add_subplot(1, 1, 1)
    for _, grp in df.groupby('run'):
        grp.plot(x_name, y_name, ax=ax, linewidth=0.5, **kwargs)

    group_by_hyper = df.groupby(x_name)
    mu_y, sem_y = group_by_hyper.mean(), group_by_hyper.agg(lambda x: x.std() / np.sqrt(x.count()))
    ax.errorbar(df[x_name].unique(), mu_y[y_name], yerr=sem_y[y_name], fmt='-k', marker='.', linewidth=1.5)
    ax.legend().remove()

    ax.set_ylabel(y_name)
    ax.set_title(f"{y_name} vs {x_name}")


if __name__ == "__main__":
    l2_model_specs = list(generate_model_specs({'dataset': 'mnist', 'task': 'sup', 'run': range(9),
                                                'l1': 0.0, 'drop': 0.0, 'l2': np.logspace(-5, -1, 9)}))
    df_l2_metrics = load_data_as_table(l2_model_specs, ['test_acc'])
    plot_by_hyper(df_l2_metrics, 'l2', 'test_acc', logx=True, figsize=(3, 2))
    plt.show()
