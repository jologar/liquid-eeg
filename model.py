import torch
import torch.nn as nn
from ncps.wirings import AutoNCP
from ncps.torch import CfC

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
    def __init__(self, seq_length, dropout):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=seq_length, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),

            nn.Conv1d(in_channels=64, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
        )
    
    def forward(self, x):
        return self.cnn(x)
    
class ConvLiquidEEG(nn.Module):
    # TODO: Prepare model for ablation tests
    def __init__(self, liquid_units=20, seq_length=100, num_classes=10, eeg_channels=5, dropout=1):
        super().__init__()
        self.last_logits = None
        self.conv_block = ConvolutionalBlock(seq_length=seq_length, dropout=dropout)
        self.liquid_block = LiquidBlock(units=liquid_units, out_features=num_classes, in_features=eeg_channels)

        self.linear = nn.Sequential(
            nn.Linear(in_features=num_classes, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(in_features=64, out_features=num_classes),
            nn.ReLU(),
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, state=None):
        x = self.conv_block(x)
        x, hx = self.liquid_block.forward(x, state)
        x = self.linear(x)
        self.last_logits = x
        return self.softmax(x), hx


class ParallelConvLiquidEEG(nn.Module):
    def __init__(self, liquid_units=20, seq_length=100, num_classes=10, eeg_channels=5, dropout=1):
        super().__init__()
        self.last_logits = None
        self.conv_block = ConvolutionalBlock(seq_length=seq_length, dropout=dropout)
        self.conv_flatten = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=10),
            nn.BatchNorm1d(10),
            nn.LeakyReLU()
        )
        self.liquid_block = LiquidBlock(units=liquid_units, out_features=6, in_features=eeg_channels)
        self.linear = nn.Sequential(
            nn.Linear(in_features=10 + 6, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(in_features=64, out_features=num_classes),
            nn.BatchNorm1d(num_classes),
            nn.LeakyReLU(),
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, state=None):
        s = self.conv_block(x)
        s_flat = self.conv_flatten(s)

        t, hx = self.liquid_block(x)

        concat = torch.cat((s_flat, t), dim=1)

        self.last_logits = self.linear(concat)
        return self.softmax(self.last_logits), hx


def count_parameters(model: ConvLiquidEEG):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
