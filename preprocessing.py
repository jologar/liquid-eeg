import torch
import torch.nn as nn

from scipy.signal import butter, sosfilt, fir_filter_design
from torch import Tensor
from constants import DEVICE

from dataset import SAMPLE_FREQ

EEG_BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 45),
}   

def butter_bandpass_sos(lowcut, highcut, freq, order = 10):
    niquist_freq = freq / 2

    low  = lowcut / niquist_freq
    high = highcut / niquist_freq
    # Butterworth filter def
    return butter(order, [low, high], btype='bandpass', output='sos')


class EEGBandsPreprocessing(nn.Module):
    def __init__(self):
        super(EEGBandsPreprocessing, self).__init__()
        self.sos_list: list = [butter_bandpass_sos(lowcut, highcut, SAMPLE_FREQ, 10) for lowcut, highcut in EEG_BANDS.values()]

    def forward(self, x: Tensor) -> Tensor:
        x = x.detach().cpu()
        band_decomposed_x = torch.cat([Tensor(sosfilt(sos=sos, x=x, axis=1)) for sos in self.sos_list], dim=-1).to(device=DEVICE)
        return band_decomposed_x



class EEGPreprocessing(nn.Module):
    def __init__(self, lowcut=1, highcut=45):
        super(EEGPreprocessing, self).__init__()
        # Butterworth filter def
        self.sos = butter_bandpass_sos(lowcut, highcut, SAMPLE_FREQ, 10)

    def forward(self, x: Tensor) -> Tensor:
        """
        A frequency filtering is done here in order to get the signal ready for prediction.
        :param x: Tensor with shape [B, L, C] with B batch dim, L sequence length dim and C EEG channels dim

        Returns the a bandwith filtered tensor with the same shape [B, L, C]
        """
        x = x.detach().cpu()
        filtered_x = sosfilt(sos=self.sos, x=x, axis=1)
        filtered_tensor = Tensor(filtered_x).to(device=DEVICE)
        return filtered_tensor