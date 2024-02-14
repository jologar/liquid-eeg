import math
import os
from typing import Any, Dict, Union
from moabb import paradigms


import numpy as np
import pandas as pd

from random import shuffle
from torcheeg import transforms
from torcheeg.datasets import BCICIV2aDataset
from torch.utils.data import Dataset, IterableDataset

BASE_BCI_C_DATASET_PATH = './datasets/bci_c'
BASE_DATASETS_PATH = './datasets/csv'
TRAIN_DS = f'{BASE_DATASETS_PATH}/train-eeg-data.csv'
VALID_DS = f'{BASE_DATASETS_PATH}/validation-eeg-data.csv'

TOTAL_FEATURES = 22
DEFAULT_FEATURES = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','A1','A2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz','X5']
SAMPLE_FREQ = 200   # 200Hz
CHUNK_MULTIPLICATOR = 50

BCI_C_IV_2A_NUM_CLASSES = 4
BCI_C_IV_2A_SAMPLE_RATE = 250
DEFAULT_BANDS = {
    'delta': [1, 4],
    'theta': [4, 8],
    'alpha': [8, 12],
    'beta': [13, 30],
    'low_gamma': [30, 40], 
}


# Custom transform to traspose matrix
class TrasposeEEG(transforms.EEGTransform):
    def __init__(self, apply_to_baseline: bool = False):
        super(TrasposeEEG, self).__init__(apply_to_baseline=apply_to_baseline)

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)
    
    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return np.moveaxis(eeg, -1, -2)


def get_bci_competition_dataset(seq_length: int, dt: int = 25, eeg_bands: dict[str, Any] = DEFAULT_BANDS) -> BCICIV2aDataset:
    bands_name = [f'{band}_{str.join(freq_range, '-')}' for band, freq_range in eeg_bands.items()]

    return BCICIV2aDataset(
        root_path='./datasets/bci_c',
        io_path=f'./datasets/bci_c/processed/biciv_2a_{seq_length}_{bands_name}',
        chunk_size=seq_length,
        overlap=seq_length - dt,
        offline_transform=transforms.BandSignal(
            sampling_rate=BCI_C_IV_2A_SAMPLE_RATE,
            band_dict=eeg_bands,
        ),
        online_transform=transforms.Compose([
            TrasposeEEG(),
            transforms.ToTensor()
        ]),
        label_transform=transforms.Compose([
                transforms.Select('label'),
                transforms.Lambda(lambda x: x - 1)
        ]),
        num_worker=6,
    )


def get_all_experiment_files() -> list[str]:
    experiment_files: list[str] = []
    for file in os.listdir(BASE_DATASETS_PATH):
        # TODO: filter only experiment csv files
        if file.endswith('.csv'):
            experiment_files.append(file)
    return experiment_files
            

class EEGIterableDataset(IterableDataset):
    def __init__(
        self,
        ds_path: str,
        target_label: str,
        sequence_length: int = 20,
        sequence_overlap: float = 0.0,
        features: list[str] = None,
        total_samples = None,
    ):
        super(EEGIterableDataset).__init__()
        self.target_label = target_label
        self.sequence_length = sequence_length
        self.sequence_overlap = sequence_overlap
        self.total_samples = total_samples
        self.csv_path = ds_path
        self.features = features if features is not None else DEFAULT_FEATURES
        self.columns = features.copy()
        self.columns.append(target_label)

        self.init_iterator()

    def init_iterator(self):
        chunksize = self.sequence_length + (self.sequence_length - self.sequence_length*self.sequence_overlap + 1)*CHUNK_MULTIPLICATOR
        self.data_iterator = pd.read_csv(self.csv_path, chunksize=chunksize, usecols=self.columns)

    def get_total_sequences(self) -> int | None:
        if self.total_samples is None: return None

        overlap = self.sequence_overlap if self.sequence_overlap > 0 else 1
        return int((self.total_samples - self.sequence_length) / (self.sequence_length * overlap))

    def get_sequence_indexes(self, sample_idx: int) -> tuple[int, int]:
        seq_idx = int(sample_idx*self.sequence_length*self.sequence_overlap + self.sequence_length)
        seq_start_idx = seq_idx - self.sequence_length
        return seq_start_idx, seq_idx - 1

    def get_stream(self):
        for chunk in self.data_iterator:
            chunk = chunk.reset_index(drop=True)

            sample_idx = 0
            seq_start_idx, seq_idx = self.get_sequence_indexes(sample_idx)
            while seq_idx < len(chunk):
                X = chunk.loc[seq_start_idx:seq_idx, chunk.columns != self.target_label].values
                y = chunk[self.target_label][seq_idx]
                sample_idx += 1
                seq_start_idx, seq_idx = self.get_sequence_indexes(sample_idx)

                yield X, y            


    def __iter__(self):
        return self.get_stream()


class CsvEEGIterableDataset(IterableDataset):
    def __init__(
        self,
        csv_files: list[str],
        target_label: str,
        sequence_length: int = 20,
        sequence_overlap: float = 0.0,
        features: list[str] = None,
    ):
        super(CsvEEGIterableDataset, self).__init__()
        self.target_label = target_label
        self.sequence_length = sequence_length
        self.sequence_overlap = sequence_overlap
        self.csv_files = csv_files
        self.features = features if features is not None else DEFAULT_FEATURES
        self.columns = features.copy()
        self.columns.append(target_label)

        self.chunksize = self.sequence_length + (self.sequence_length - self.sequence_length*self.sequence_overlap + 1)*CHUNK_MULTIPLICATOR

    def get_sequence_indexes(self, sample_idx: int) -> tuple[int, int]:
        seq_idx = int(sample_idx*self.sequence_length*self.sequence_overlap + self.sequence_length)
        seq_start_idx = seq_idx - self.sequence_length
        return seq_start_idx, seq_idx - 1

    def get_stream(self):
        for file in self.csv_files:
            chunk = pd.read_csv(f'{BASE_DATASETS_PATH}/{file}')
            sample_idx = 0
            seq_start_idx, seq_idx = self.get_sequence_indexes(sample_idx)
            while seq_idx < len(chunk):
                X = chunk.loc[seq_start_idx:seq_idx, chunk.columns != self.target_label].values
                y = chunk[self.target_label][seq_idx]
                
                sample_idx += 1
                seq_start_idx, seq_idx = self.get_sequence_indexes(sample_idx)

                yield X, y            

    def __iter__(self):
        return self.get_stream()


class EEGDataset(Dataset):
    def __init__(
        self,
        target_label: str,
        pandas_df: pd.DataFrame | None = None,
        csv_files: list[str] | None = None,
        sequence_length: int = 20,
        sequence_overlap: float = 0.0,
        features: list[str] | None = None, 
    ) -> None:
        if csv_files and pandas_df:
            raise ValueError('pandas_df or csv_file should be pased but NOT both.')
        if csv_files is None and pandas_df is None:
            raise ValueError('panas_df or csv_file MUST be passed.')
        if csv_files is not None and len(csv_files) < 1:
            raise ValueError('A minimum of 1 csv file MUST be passed.')
        if csv_files is not None:
            # Load and format the csv dataset
            df_list = []
            for file in csv_files:
                df = pd.read_csv(file)
                df = df.sort_values(by=df.columns[0])
                df = df.drop(df.columns[0], axis=1)
                df_list.append(df)
            df = pd.concat(df_list)
            df = df.reset_index(drop=True)
            self.eeg_data = df
            del df

        if pandas_df is not None:
            self.eeg_data = pandas_df
            self.eeg_data = self.eeg_data.reset_index(drop=True)

        self.sequence_overlap = sequence_overlap
        self.target_label = target_label
        self.features = features

        if self.features is None:
            self.features = list(self.eeg_data.columns)
            self.features.remove(target_label)

        self.sequence_length = sequence_length

    def __len__(self) -> int:
        overlap = self.sequence_overlap if self.sequence_overlap > 0 else 1
        return int((len(self.eeg_data) - self.sequence_length) / (self.sequence_length * overlap))
    
    def __getitem__(self, index: int) -> tuple:
        index = int(index*self.sequence_length*self.sequence_overlap + self.sequence_length)
        seq_start_idx = index - self.sequence_length

        X = self.eeg_data[self.features][seq_start_idx:index].values
        y = self.eeg_data[self.target_label][index]
        return X, y
    
    def split(self, split_ratio: float = 0.8) -> tuple:
        split_idx = int(len(self.eeg_data) * split_ratio)
        train_ds = EEGDataset(
            target_label=self.target_label,
            pandas_df=self.eeg_data[:split_idx],
            sequence_length=self.sequence_length,
            sequence_overlap=self.sequence_overlap,
            features=self.features,
        )
        valid_ds = EEGDataset(
            target_label=self.target_label,
            pandas_df=self.eeg_data[split_idx+1:],
            sequence_length=self.sequence_length,
            sequence_overlap=self.sequence_overlap,
            features=self.features,
        )
        
        return train_ds, valid_ds
        
class KFoldSplitDataset(Dataset):
    def __init__(
            self,
            csv_files: list[str],
            target_label: str,
            k_folds: int = 10,
            shuffle_files: bool = True,
            sequence_length: int = 20,
            sequence_overlap: float = 0.0,
            features: list[str] = None,
        ):
        super(KFoldSplitDataset, self).__init__()
        if shuffle_files:
            shuffle(csv_files)

        self.csv_files = csv_files
        self.target_label = target_label
        self.sequence_length = sequence_length
        self.sequence_overlap = sequence_overlap
        self.features = features
        self.test_fold_size = math.ceil(len(self.csv_files) // k_folds)
        self.k_folds = k_folds
        print(f'>>>>>>>>>>>> CSV FILES: {len(self.csv_files)} test fold: {self.test_fold_size}')

    def __len__(self) -> int:
        return self.k_folds
    
    def __getitem__(self, index) -> tuple[IterableDataset, IterableDataset]:
        valid_fold_start = index 
        valid_fold_end = valid_fold_start + self.test_fold_size
        
        valid_files = self.csv_files[valid_fold_start:valid_fold_end]
        train_files = [file for idx, file in enumerate(self.csv_files) if idx not in range(valid_fold_start, valid_fold_end)]

        train_ds = CsvEEGIterableDataset(
            csv_files=train_files,
            target_label=self.target_label,
            sequence_length=self.sequence_length,
            sequence_overlap=self.sequence_overlap,
            features=self.features
        )

        valid_ds = CsvEEGIterableDataset(
            csv_files=valid_files,
            target_label=self.target_label,
            sequence_length=self.sequence_length,
            sequence_overlap=self.sequence_overlap,
            features=self.features
        )
        
        return train_ds, valid_ds
