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
    HIDDEN_LAYERS = (1, 2)

    def __init__(self, pdrop=0.0):
        super().__init__()
        self.pdrop = pdrop

        self.fc1 = nn.Linear(INPUT_DIM, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, CLASSES)

        self.layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x, depth=3):
        single_output = not is_iterable(depth)
        if single_output:
            depth = [depth]
        outputs = []
        h = x.view(x.size(0), -1)
        for i, l in enumerate(self.layers):
            if i in depth:
                outputs.append(h)
            h = F.dropout(F.relu(l(h)), p=self.pdrop, training=self.training)
        if len(self.layers) in depth:
            outputs.append(h)
        return outputs[0] if single_output else outputs


class MnistAutoEncoder(nn.Module):

    DATASET = 'mnist'
    TASK = 'unsupervised'
    HIDDEN_LAYERS = (1,)

    def __init__(self, h_dim=32, pdrop=0.0):
        super().__init__()

        self.pdrop = pdrop

        # Encoder
        self.fc1 = nn.Linear(INPUT_DIM, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, h_dim)

        # Decoder
        self.fc4 = nn.Linear(h_dim, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, INPUT_DIM)

        self.enc_layers = [self.fc1, self.fc2, self.fc3]
        self.dec_layers = [self.fc4, self.fc5, self.fc6]

    def forward(self, x, depth=2):
        single_output = not is_iterable(depth)
        if single_output:
            depth = [depth]
        outputs = []
        h = x.view(x.size(0), -1)
        if 0 in depth:
            outputs.append(h)
        for i, l in enumerate(self.enc_layers):
            h = F.dropout(F.relu(l(h)), p=self.pdrop, training=self.training)
        if 1 in depth:
            outputs.append(h)
        for i, l in enumerate(self.dec_layers):
            h = F.dropout(F.relu(l(h)), p=self.pdrop, training=self.training)
        if 2 in depth:
            outputs.append(h)
        return outputs[0] if single_output else outputs
