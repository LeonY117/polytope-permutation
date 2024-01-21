from torch import nn
from typing import List


class DQN(nn.Module):
    def __init__(self, inp: int, oup: int, units: List[int]):
        # TODO: we can optionally add normalization layers
        super().__init__()
        layers = []
        prev = inp
        for num_units in units:
            layers.append(nn.Linear(prev, num_units))
            layers.append(nn.ReLU())
            prev = num_units

        layers.append(nn.Linear(units[-1], oup))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
