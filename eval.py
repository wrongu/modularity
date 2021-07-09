import torch
import torch.nn.functional as F
from models import LitWrapper
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

    # TODO - add modularity scores

    torch.save(info, checkpoint_file)
    return info


if __name__ == '__main__':
    from pprint import pprint
    from pathlib import Path
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--ckpt-file', metavar='CKPT', type=Path)
    parser.add_argument('--data-dir', metavar='DATA', type=Path)
    parser.add_argument('--metrics', default=None)
    args = parser.parse_args()

    if args.metrics is not None:
        args.metrics = [s.split() for s in args.metrics.split(",")]

    info = evaluate(args.ckpt_file, args.data_dir, args.metrics)
    pprint(info)
