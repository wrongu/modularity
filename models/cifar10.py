import torch
import torch.nn as nn
import torch.nn.functional as F


# Some global CIFAR metadata
INPUT_SIZE = (3, 32, 32)
INPUT_DIM = 3*32*32
CLASSES = 10


def validate_layer_size(layer, in_size):
    return layer(torch.randn((1,) + in_size)).size()[1:]


def prod(vals):
    out = 1
    for v in vals:
        out *= v
    return out


class ResBlock(nn.Module):
    """Residual block, inspired by res_block in https://github.com/davidcpage/cifar10-fast/
    """
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu1 = nn.ReLU(True)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu2 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)

        if (stride != 1) or (c_out != c_in):
            self.proj = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.proj = lambda x: x

    def forward(self, x):
        # Preprocess with relu(batchnorm(x))
        x = self.relu1(self.bn1(x))
        # Compute the branch
        branch = self.conv2(self.relu2(self.bn2(self.conv1(x))))
        # Add branch to (maybe) projected input
        return self.proj(x) + branch


class ClassifierPool(nn.Module):
    """Optionally do max or [max, avg] pooling, inspired by https://github.com/davidcpage/cifar10-fast/
    """
    def __init__(self, concat=True):
        super(ClassifierPool, self).__init__()
        self.concat = concat
        self.max = nn.MaxPool2d(4)
        self.avg = nn.AvgPool2d(4)

    def forward(self, x):
        if self.concat:
            # Concatenate both max- and avg-pool along channels dimension (assume x is size (batch, channel, h, w))
            return torch.cat([self.max(x), self.avg(x)], dim=1)
        else:
            # Just do max-pooling
            return self.max(x)


class Cifar10Fast(nn.Module):
    """A good architecture for CIFAR10, inspired by https://github.com/davidcpage/cifar10-fast/
    """

    DATASET = 'cifar10'
    TASK = 'supervised'

    def __init__(self, channels=(64, 128, 256, 256), concat_pool=True):
        super(Cifar10Fast, self).__init__()

        # Sizes: (3, 32, 32) --> (channels[0], 32, 32)
        self.conv1 = nn.Conv2d(INPUT_SIZE[0], channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu1 = nn.ReLU(True)
        sz = validate_layer_size(self.conv1, INPUT_SIZE)


        # Sizes: (64, 32, 32) --> (64, 32, 32)
        self.layer1 = nn.Sequential(
            ResBlock(channels[0], channels[0], 1),
            ResBlock(channels[0], channels[0], 1)
        )
        sz = validate_layer_size(self.layer1, sz)

        # Sizes: (64, 32, 32) --> (128, 16, 16)
        self.layer2 = nn.Sequential(
            ResBlock(channels[0], channels[1], 2),
            ResBlock(channels[1], channels[1], 1)
        )
        sz = validate_layer_size(self.layer2, sz)

        # Sizes: (128, 16, 16) --> (256, 8, 8)
        self.layer3 = nn.Sequential(
            ResBlock(channels[1], channels[2], 2),
            ResBlock(channels[2], channels[2], 1)
        )
        sz = validate_layer_size(self.layer3, sz)

        # Sizes: (256, 8, 8) --> (256, 4, 4)
        self.layer4 = nn.Sequential(
            ResBlock(channels[2], channels[3], 2),
            ResBlock(channels[3], channels[3], 1)
        )
        sz = validate_layer_size(self.layer4, sz)

        # Sizes: (256, 4, 4) --> (512, 1, 1) if concat_pool else (256, 1, 1)
        self.pool = ClassifierPool(concat=concat_pool)
        sz = validate_layer_size(self.pool, sz)

        # Project to logits
        self.proj = nn.Linear(prod(sz), CLASSES)

    def forward(self, x):
        batches = x.size(0)
        h0 = self.relu1(self.bn1(self.conv1(x)))
        h1 = self.layer1(h0)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        h5 = self.pool(h4)
        return [h0, h1, h2, h3, h4, h5], self.proj(h5.view(batches, -1))
