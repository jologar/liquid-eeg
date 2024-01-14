import torch
import torch.nn as nn
from ncps.wirings import AutoNCP
from ncps.torch import CfC
from constants import DEVICE

from preprocessing import EEGBandsPreprocessing, EEG_BANDS

class LiquidBlock(nn.Module):
    def __init__(self, units=20, out_features=10, in_features=5):
        super().__init__()
        wiring = AutoNCP(units, out_features)
        self.units = units
        self.liquid = CfC(in_features, wiring, return_sequences=False, batch_first=True)

    def forward(self, x, state=None):
        x, hx = self.liquid.forward(input=x, hx=state)
        return x, hx
    

class ConvolutionalBlock(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.cnn = nn.Sequential(
            # nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # # nn.Dropout2d(p=dropout),
            # nn.MaxPool2d(kernel_size=3, stride=2),

            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # # nn.Dropout2d(p=dropout),
            # nn.MaxPool2d(kernel_size=3, stride=2),

            # nn.Conv2d(in_channels=32, out_channels=1, kernel_size=2, stride=1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(),
            # nn.Dropout2d(p=dropout)

            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.cnn(x)
    
class ConvLiquidEEG(nn.Module):
    # TODO: Prepare model for ablation tests
    def __init__(self, liquid_units=20, seq_length=100, num_classes=10, eeg_channels=5, dropout=1):
        super().__init__()
        self.last_logits = None
        self.preprocessing = EEGBandsPreprocessing()
        self.conv_block = ConvolutionalBlock(dropout=dropout)
        self.liquid_block = LiquidBlock(units=liquid_units, out_features=num_classes, in_features=8)

        # self.linear = nn.Sequential(
        #     nn.LazyLinear(in_features=num_classes, out_features=num_classes),
        #     nn.ReLU(),
        # )
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, x, state=None):
        # print(f'>>>>>>> INPUT: {x.shape}')
        x = self.preprocessing(x)
        # print(f'>>>>>>>>> PRE OUTPUT: {x.shape}')
        x = self.conv_block(x.unsqueeze(1))
        # print(f'>>>>>>>>>> CONV OUTPUT SHAPE: {x.shape}')
        x, hx = self.liquid_block.forward(torch.squeeze(x), state)
        # print(f'>>>>>>>>> LIQUID OUT: {x.shape}')
        # x = self.linear(x)
        self.last_logits = x
        return self.softmax(x), hx


class ParallelConvLiquidEEG(nn.Module):
    def __init__(self, liquid_units=20, seq_length=100, num_classes=10, eeg_channels=5, dropout=1):
        super().__init__()
        self.last_logits = None
        
        self.preprocessing = EEGBandsPreprocessing()
        self.conv_block = ConvolutionalBlock(seq_length=seq_length, dropout=dropout)
        self.conv_flatten = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
        )
        sensory_units = eeg_channels * len(EEG_BANDS)
        self.liquid_block = LiquidBlock(units=liquid_units, out_features=32, in_features=sensory_units)
        self.linear = nn.Sequential(
            nn.LazyLinear(out_features=num_classes),
            nn.BatchNorm1d(num_classes),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, state=None):
        x = self.preprocessing(x)
        l: torch.Tensor = x.detach()
        s = self.conv_block(x.unsqueeze(1))
        s_flat = self.conv_flatten(s)
        l.requires_grad = True
        t, hx = self.liquid_block(l.to(device=DEVICE))

        concat = torch.cat((s_flat, t), dim=1)
        self.last_logits = self.linear(concat)
        return self.softmax(self.last_logits), hx


def count_parameters(model: ConvLiquidEEG):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
