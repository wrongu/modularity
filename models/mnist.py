import torch.nn as nn
import torch.nn.functional as F


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
        x = x.view(x.size(0), -1)
        h0 = F.dropout(F.relu(self.fc1(x)), p=self.pdrop, training=self.training)
        h1 = F.dropout(F.relu(self.fc2(h0)), p=self.pdrop, training=self.training)
        return [h0, h1], self.fc3(h1)


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
        x = x.view(x.size(0), -1)
        # Encoder stage
        e0 = F.dropout(F.relu(self.fc1(x)), p=self.pdrop, training=self.training)
        e1 = F.dropout(F.relu(self.fc2(e0)), p=self.pdrop, training=self.training)
        # Bottleneck stage -- no relu or dropout
        h = self.fc3(e1)
        # Decoder stage
        d0 = F.dropout(F.relu(self.fc4(h)), p=self.pdrop, training=self.training)
        d1 = F.dropout(F.relu(self.fc5(d0)), p=self.pdrop, training=self.training)
        return [h], self.fc6(d1)
