import torch
import torchvision
from torch.utils.data import random_split
from .mnist import MnistSupervised, MnistAutoEncoder


def get_dataset(dataset, data_dir, train_val_split, seed=286436723):
    # Note: the default seed was chosen from a single call to random.randint
    if dataset.lower() == 'mnist':
        trans = torchvision.transforms.ToTensor()
        train = torchvision.datasets.MNIST(data_dir / 'mnist', train=True, transform=trans)
        test = torchvision.datasets.MNIST(data_dir / 'mnist', train=False, transform=trans)
    else:
        raise ValueError(f"Unrecognized dataset {dataset}")

    n_train = int(len(train)*train_val_split)
    n_val = len(train) - n_train
    train, val = random_split(train, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    return train, val, test


def get_model(dataset, task, pdrop):
    if dataset.lower() == 'mnist' and task.lower()[:3] == 'sup':
        return MnistSupervised()
    elif dataset.lower() == 'mnist' and task.lower()[:5] == 'unsup':
        return MnistAutoEncoder(pdrop=pdrop)
    else:
        raise ValueError(f"Unrecognized dataset x task combo: {dataset} x {task}")


def get_uid(**kwargs):
    uid = f"{kwargs['dataset']}_{kwargs['task']}"
    optional_hypers = ['l2', 'l1', 'drop', 'run']
    for h in optional_hypers:
        if h not in kwargs:
            continue
        val = kwargs[h]
        if val > 0:
            if isinstance(val, int):
                uid += f"_{h}={val}"
            else:
                uid += f"_{h}={val:.5f}"
    return uid