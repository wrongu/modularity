import torch.nn as nn
import torch.nn.functional as F
from util import is_iterable


# Some global MNIST metadata
INPUT_SIZE = (1, 28, 28)
INPUT_DIM = 1*28*28
CLASSES = 10


class MnistSupervised(nn.Module):

    DATASET = 'mnist'
    TASK = 'supervised'
    HIDDEN_DIMS = (64, 64)

    def __init__(self, pdrop=0.0):
        super().__init__()
        self.pdrop = pdrop

        self.fc1 = nn.Linear(INPUT_DIM, MnistSupervised.HIDDEN_DIMS[0])
        self.fc2 = nn.Linear(MnistSupervised.HIDDEN_DIMS[0], MnistSupervised.HIDDEN_DIMS[1])
        self.fc3 = nn.Linear(MnistSupervised.HIDDEN_DIMS[1], CLASSES)

        self.layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        hidden = []
        h = x.view(x.size(0), -1)
        for i, l in enumerate(self.layers):
            hidden.append(h)
            h = F.dropout(F.relu(l(h)), p=self.pdrop, training=self.training)
        return hidden, h


class MnistAutoEncoder(nn.Module):

    DATASET = 'mnist'
    TASK = 'unsupervised'
    HIDDEN_DIMS = (32,)

    def __init__(self, pdrop=0.0):
        super().__init__()

        self.pdrop = pdrop

        # Encoder
        self.fc1 = nn.Linear(INPUT_DIM, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, MnistAutoEncoder.HIDDEN_DIMS[0])

        # Decoder
        self.fc4 = nn.Linear(MnistAutoEncoder.HIDDEN_DIMS[0], 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, INPUT_DIM)

        self.enc_layers = [self.fc1, self.fc2, self.fc3]
        self.dec_layers = [self.fc4, self.fc5, self.fc6]

    def forward(self, x):
        h = x.view(x.size(0), -1)
        for i, l in enumerate(self.enc_layers):
            h = F.dropout(F.relu(l(h)), p=self.pdrop, training=self.training)
        hidden = [h]
        for i, l in enumerate(self.dec_layers):
            h = F.dropout(F.relu(l(h)), p=self.pdrop, training=self.training)
        return hidden, h
