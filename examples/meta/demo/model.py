import torch
from torch import nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for x in range(8):
            layers.append(nn.Linear(4, 4))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, input, target):
        output = self.layers(input)
        return torch.nn.functional.cross_entropy(output, target)
