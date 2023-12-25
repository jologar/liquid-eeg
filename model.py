from collections import OrderedDict
import torch.nn as nn
import numpy as np
from ncps.wirings import AutoNCP
from ncps.torch import CfC


def convolutional_layer(layer: int, kernels: int = 64, kernel_size: int = 5, dropout: float = 0.0):
    padding = kernel_size // 2

    return [
        nn.Conv2d(in_channels=1, out_channels=kernels, kernel_size=kernel_size, stride=1, padding=padding),
        nn.BatchNorm2d(kernels),
        nn.ReLU(),
        nn.Dropout2d(p=dropout),
        nn.MaxPool2d(kernel_size=3, stride=2),
    ]


class LiquidBlock(nn.Module):
    def __init__(self, units=20, out_features=10, in_features=5):
        super().__init__()
        wiring = AutoNCP(units, out_features)
        self.units = units
        self.liquid = CfC(in_features, wiring, return_sequences=True, batch_first=True)

    def forward(self, x, state=None):
        x, hx = self.liquid.forward(input=x, hx=state)
        return x, hx
    

class ConvolutionalBlock(nn.Module):
    def __init__(self, num_layers, out_features, dropout, kernels):
        super().__init__()
        
        layers = [convolutional_layer(idx, kernels=kernels, kernel_size=3, dropout=dropout) for idx in range(num_layers)]
        layers = list(np.array(layers).flatten())

        self.cnn = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.LazyLinear(out_features=out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
        )
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x
    
class LiquidEEG(nn.Module):
    # TODO: Prepare model for ablation tests
    def __init__(self, liquid_units=20, num_classes=10, sensory_units=5, dropout=0.0):
        super().__init__()
        self.conv_block = ConvolutionalBlock(num_layers=3, out_features=sensory_units, dropout=dropout, kernels=64)
        self.liquid_block = LiquidBlock(units=liquid_units, out_features=num_classes, in_features=sensory_units)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, state=None):
        x = self.conv_block(x)
        x, hx = self.liquid_block.forward(x, state)
        return self.softmax(x), hx


def count_parameters(model: LiquidEEG):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
