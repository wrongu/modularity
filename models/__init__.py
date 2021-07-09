import torch
import torchvision
from torch.utils.data import random_split
from .mnist import MnistSupervised, MnistAutoEncoder
import pytorch_lightning as pl
import torch.nn.functional as F


class LitWrapper(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # Assert minimum arguments were given
        required_args = ['dataset', 'task', 'l1', 'l2', 'drop']
        for arg in required_args:
            if arg not in kwargs:
                raise RuntimeError(f"Missing argument to LitWrapper: {arg}")

        # Add default arguments if not provided using `dict.get(key) or default` notation, which sets the value
        # stored in 'key' if it is missing (get returns None) or overrides it if it was None or falsey (0, 0., etc)
        kwargs['run'] = kwargs.get('run') or 0
        # By default, seed changes depending on run, but constant across other parameter changes
        kwargs['seed'] = kwargs.get('seed') or (286436723 + kwargs['run'])
        kwargs['train_val_split'] = kwargs.get('train_val_split') or 0.9

        # Copy all kwargs fields into instance variables and log them into self.hparams
        self.save_hyperparameters(kwargs)

    def init_model(self, set_seed=True):
        if set_seed:
            pl.seed_everything(self.hparams.seed)

        # Select among architectures based on the given dataset, task, etc
        if self.hparams.dataset.lower() == 'mnist' and self.hparams.task.lower()[:3] == 'sup':
            self.model = MnistSupervised(pdrop=self.hparams.drop)
            self.dataset, self.hparams.task = 'mnist', 'sup'
        elif self.hparams.dataset.lower() == 'mnist' and self.hparams.task.lower()[:5] == 'unsup':
            self.model = MnistAutoEncoder(pdrop=self.hparams.drop)
            self.dataset, self.hparams.task = 'mnist', 'unsup'
        else:
            raise ValueError(f"Unrecognized dataset x task combo: {self.hparams.dataset} x {self.hparams.task}")

    def on_load_checkpoint(self, ckpt_file):
        # Instantiate self.model, but let the calling function handle state_dict stuff
        self.init_model(set_seed=False)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def _step(self, batch):
        x, y = batch
        _, out = self(x)
        if self.hparams.task[:3] == 'sup':
            loss = F.cross_entropy(out, y)
        else:
            loss = F.mse_loss(out, x.view(x.size(0), -1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss)
        return loss + self.hparams.l1 * self.l1_norm() + self.hparams.l2 * self.l2_norm()

    def on_epoch_end(self):
        self.log('l1_norm', self.l1_norm())
        self.log('l2_norm', self.l2_norm())

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        # TODO - is this accumulating? Would this affect early stopping? Or resuming from checkpoints?
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def l1_norm(self):
        return sum(p.abs().sum() for p in self.model.parameters() if p.ndim >= 2)

    def l2_norm(self):
        return sum((p**2).sum() for p in self.model.parameters() if p.ndim >= 2)

    def get_dataset(self, data_dir):
        if self.hparams.dataset.lower() == 'mnist':
            trans = torchvision.transforms.ToTensor()
            train = torchvision.datasets.MNIST(data_dir / 'mnist', train=True, transform=trans)
            test = torchvision.datasets.MNIST(data_dir / 'mnist', train=False, transform=trans)
        else:
            raise ValueError(f"Unrecognized dataset {self.hparams.dataset}")

        n_train = int(len(train)*self.hparams.train_val_split)
        n_val = len(train) - n_train
        train, val = random_split(train, [n_train, n_val], generator=torch.Generator().manual_seed(self.hparams.seed))
        return train, val, test

    def get_uid(self):
        uid = f"{self.hparams.dataset}_{self.hparams.task}"
        optional_hypers = ['l2', 'l1', 'drop', 'run']
        for h in optional_hypers:
            if h not in self.hparams:
                continue
            val = self.hparams[h]
            if val:
                if isinstance(val, int):
                    uid += f"_{h}={val}"
                else:
                    uid += f"_{h}={val:.5f}"
        return uid