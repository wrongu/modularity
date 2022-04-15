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

