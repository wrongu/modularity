import torch.nn as nn
import torch.nn.functional as F


# Some global MNIST metadata
INPUT_SIZE = (1, 28, 28)
INPUT_DIM = 1*28*28
CLASSES = 10


class MnistSupervised(nn.Module):

    DATASET = 'mnist'
    TASK = 'supervised'

    def __init__(self, pdrop=0.0, channels=(64, 64)):
        super().__init__()
        self.pdrop = pdrop

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(INPUT_DIM, channels[0]))
        for d1, d2 in zip(channels[:-1], channels[1:]):
            self.fc_layers.append(nn.Linear(d1, d2))
        self.proj = nn.Linear(channels[-1], CLASSES)

    def forward(self, x):
        # Flatten
        x = x.view(x.size(0), -1)
        # Apply layers in order
        hidden = []
        for lay in self.fc_layers:
            x = F.dropout(F.relu(lay(x)), p=self.pdrop, training=self.training)
            hidden.append(x)
        return hidden, self.proj(x)

