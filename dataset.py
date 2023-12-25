import pandas as pd
import torch

from torch.utils.data import Dataset

TOTAL_FEATURES = 22

# Define custom dataset to load sequences
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
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(index*self.sequence_length*self.sequence_overlap + self.sequence_length)
        seq_start_idx = index - self.sequence_length

        X = torch.tensor(self.eeg_data[self.features][seq_start_idx:index].values).type(torch.FloatTensor)
        y = torch.tensor(self.eeg_data[self.target_label][index]).type(torch.LongTensor)
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