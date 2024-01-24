from torch import nn
from torchrl.modules import NoisyLinear
from typing import List
from functools import partial


class DQN(nn.Module):
    def __init__(self, inp: int, oup: int, units: List[int], noisy: bool = False):
        # TODO: we can optionally add normalization layers
        super().__init__()
        layers = []
        prev = inp
        layer = partial(NoisyLinear, std_init=0.5) if noisy else nn.Linear
        for num_units in units:
            layers.append(layer(prev, num_units))
            layers.append(nn.ReLU())
            prev = num_units

        layers.append(layer(units[-1], oup))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
