import torch
import torchvision
from torch.utils.data import random_split
from .mnist import MnistSupervised
from .cifar10 import Cifar10Fast
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

    def init_model(self, set_seed=True, **extra_model_args):
        if set_seed:
            pl.seed_everything(self.hparams.seed)

        # Select among architectures based on the given dataset, task, etc
        if self.hparams.dataset.lower() == 'mnist' and self.hparams.task.lower()[:3] == 'sup':
            self.model = MnistSupervised(pdrop=self.hparams.drop, **extra_model_args)
            self.dataset, self.hparams.task = 'mnist', 'sup'
            self.loss_fn = lambda x, y, out: F.cross_entropy(out, y)
            input_size = (1, 28, 28)
        elif self.hparams.dataset.lower() == 'cifar10' and self.hparams.task.lower()[:3] == 'sup':
            self.model = Cifar10Fast(pdrop=self.hparams.drop, **extra_model_args)
            self.dataset, self.hparams.task = 'cifar10', 'sup'
            self.loss_fn = lambda x, y, out: F.cross_entropy(out, y)
            input_size = (3, 32, 32)
        else:
            raise ValueError(f"Unrecognized dataset x task combo: {self.hparams.dataset} x {self.hparams.task}")

        # Discover hidden activity sizes
        dummy_input = torch.randn(input_size)
        hidden, _ = self.model(dummy_input.unsqueeze(0))
        self.hidden_dims = [h.size()[1:] for h in hidden]

    def on_load_checkpoint(self, ckpt_data):
        # Instantiate self.model, but let the calling function handle state_dict stuff
        self.init_model(set_seed=False, **ckpt_data['hyper_parameters'].get('model_args', {}))

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        sched = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.25, patience=3),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'name': 'LR'
        }
        return [opt], [sched]

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, out = self(x)
        loss = self.loss_fn(x, y, out)
        l1_norm = self.l1_norm()
        l2_norm = self.l2_norm()
        acc = torch.mean((torch.argmax(out, dim=1) == y).float())
        self.log('train_loss', loss)
        self.log('l1_norm', l1_norm)
        self.log('l2_norm', l2_norm)
        self.log('train_acc', acc)
        return loss + self.hparams.l1 * l1_norm + self.hparams.l2 * l2_norm

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, out = self(x)
        loss = self.loss_fn(x, y, out)
        acc = torch.mean((torch.argmax(out, dim=1) == y).float())
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        return loss

    def l1_norm_by_layer(self):
        return [p.abs().sum() for p in self.model.parameters() if p.ndim >= 2]

    def l1_norm(self):
        return sum(self.l1_norm_by_layer())

    def l2_norm_by_layer(self):
        return [(p**2).sum() for p in self.model.parameters() if p.ndim >= 2]

    def l2_norm(self):
        return sum(self.l2_norm_by_layer())

    def nuc_norm_by_layer(self):
        return [torch.norm(p, p="nuc") for p in self.model.parameters() if p.ndim == 2]

    def nuc_norm(self):
        return sum(self.nuc_norm_by_layer())

    def sparsity(self, eps=1e-3):
        return torch.mean((torch.cat([p.flatten() for p in self.model.parameters() if p.ndim >= 2], dim=0).abs() < eps).float())

    def sparsity_by_layer(self, eps=1e-3):
        return [torch.mean((p.flatten().abs() < eps).float()) for p in self.model.parameters() if p.ndim >= 2]

    def get_dataset(self, data_dir):
        if self.hparams.dataset.lower() == 'mnist':
            trans = torchvision.transforms.ToTensor()
            train = torchvision.datasets.MNIST(data_dir / 'mnist', train=True, transform=trans)
            test = torchvision.datasets.MNIST(data_dir / 'mnist', train=False, transform=trans)
        elif self.hparams.dataset.lower() == 'cifar10':
            train_trans = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomHorizontalFlip()
            ])
            trans = torchvision.transforms.ToTensor()
            train = torchvision.datasets.CIFAR10(data_dir / 'cifar10', train=True, transform=train_trans)
            test = torchvision.datasets.CIFAR10(data_dir / 'cifar10', train=False, transform=trans)
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
                elif val >= 0.00001:
                    uid += f"_{h}={val:.5f}"
                else:
                    uid += f"_{h}={val:.2e}"
        return uid