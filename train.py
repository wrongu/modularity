import torch
import argparse
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from models import get_dataset, get_model, get_uid
from torch.utils.data import DataLoader
from pathlib import Path
from sys import exit


class LitWrapper(pl.LightningModule):
    def __init__(self, hprs:argparse.Namespace):
        super().__init__()

        # Copy all fields of args into instance variables and log them into hparams
        self.save_hyperparameters(hprs)

        # Rely on 'models' module for logic of selecting an architecture for a dataset x task
        self.model = get_model(self.hparams.dataset, self.hparams.task, self.hparams.drop)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        if self.hparams.task[:3] == 'sup':
            loss = F.cross_entropy(out, y)
        else:
            loss = F.mse_loss(out, x.view(x.size(0), -1))
        self.log('train_loss', loss)
        return loss + self.hparams.l1 * self.l1_norm() + self.hparams.l2 * self.l2_norm()

    def on_epoch_end(self):
        self.log('l1_norm', self.l1_norm())
        self.log('l2_norm', self.l2_norm())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        if self.hparams.task[:3] == 'sup':
            loss = F.cross_entropy(out, y)
        else:
            loss = F.mse_loss(out, x.view(x.size(0), -1))
        # TODO - is this accumulating? Would this affect early stopping? Or resuming from checkpoints?
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def l1_norm(self):
        return sum(p.abs().sum() for p in self.model.parameters() if p.ndim >= 2)

    def l2_norm(self):
        return sum((p**2).sum() for p in self.model.parameters() if p.ndim >= 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Trainer config
    parser.add_argument('--device', metavar='DEV', default='auto')
    parser.add_argument('--devices', metavar='CUDA_IDS', default='0123')
    parser.add_argument('--train-val-split', default=0.9, type=float)
    parser.add_argument('--workers', default=2, type=int)
    # Model config
    parser.add_argument('--dataset', metavar='DAT', type=str)
    parser.add_argument('--task', metavar='TSK', type=str)
    parser.add_argument('--l2', default=1e-5, type=float)
    parser.add_argument('--l1', default=0.0, type=float)
    parser.add_argument('--drop', default=0.0, type=float)
    parser.add_argument('--run', default=0, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--seed', default=None)
    # Environment config
    parser.add_argument('--save-dir', metavar='DIR', required=True, type=Path)
    parser.add_argument('--data-dir', default='data', type=Path)
    parser.add_argument('--batch-size', default=200, type=int)
    args = parser.parse_args()

    if args.seed is None:
        # Each 'run' is a different RNG, but the same 'run' across hyperparameter values will use the same RNG.
        # The baseline value to which run is added was chosen by a single call to random.randint(0, 2**32)
        args.seed = 286436723 + args.run

    # TODO - verify that checkpoint-loading respects RNG state
    weights_dir = Path(args.save_dir) / get_uid(**args.__dict__) / 'weights'
    the_checkpoint = weights_dir / 'last.ckpt'
    if not the_checkpoint.exists():
        the_checkpoint = None
    else:
        info = torch.load(the_checkpoint)
        if info['epoch'] >= args.epochs:
            print(f"Nothing to do – model is trained up to {args.epochs} epochs already!")
            exit(0)

    # pytorch-lightning's seed_everything supposedly takes care of cuda, torch, python, etc...
    pl.seed_everything(args.seed)
    # ...but we also have to tell cudnn to do the same thing every time by (i) not running benchmarks internally...
    torch.backends.cudnn.benchmark = False
    # ... and (ii) running everything in 'deterministic mode'
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    the_gpu = None
    if args.device == 'auto' and torch.cuda.is_available():
        avail_gpus = [int(d) for d in args.devices]
        the_gpu = [avail_gpus[args.run % len(avail_gpus)]]
        print("AUTOMATICALLY SELECTING GPU:", the_gpu)
    elif args.device in '0123456789':
        the_gpu = [int(args.device)]

    train, val, test = get_dataset(args.dataset, args.data_dir, args.train_val_split, seed=args.seed)
    pl_model = LitWrapper(args)
    cb = [pl.callbacks.ModelCheckpoint(dirpath=weights_dir, monitor='val_loss', save_last=True, save_top_k=5)]
    tblogger = TensorBoardLogger(args.save_dir, name=get_uid(**args.__dict__), version=0)
    # Debug - log info to ensure the train/val/test splits are identical for a given run
    tblogger.experiment.add_image('train_0', train[0][0])
    tblogger.experiment.add_image('val_0', val[0][0])
    tblogger.experiment.add_image('test_0', test[0][0])
    trainer = pl.Trainer(logger=tblogger,
                         callbacks=cb,
                         deterministic=True,
                         resume_from_checkpoint=the_checkpoint,
                         default_root_dir=args.save_dir,
                         gpus=the_gpu,
                         auto_select_gpus=False,
                         max_epochs=args.epochs)
    trainer.fit(pl_model,
                train_dataloader=DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                            pin_memory=True, num_workers=args.workers),
                val_dataloaders=DataLoader(val, batch_size=args.batch_size,
                                           pin_memory=True, num_workers=args.workers))