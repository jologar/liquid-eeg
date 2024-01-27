import glob
import json
import os
import random
import time

import numpy as np
import torch

from pydantic import BaseModel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, IterableDataset
from constants import DEVICE

from dataset import TOTAL_FEATURES, CsvEEGIterableDataset, KFoldSplitDataset, get_all_experiment_files
from model import ConvLiquidEEG, count_parameters
from training import EPOCHS, val_loop, train_loop

NUM_CLASSES = 4
LOG_BASE_DIR = './log/experiments'

class ExperimentConfig(BaseModel):
    name: str
    target_label: str
    learning_rate: float
    decay_rate:float
    liquid_units: int
    dropout: float
    batch_size: int
    sequence_length: int
    sequence_overlap: float
    features: list[str] | None = None


class Experiment:
    def __init__(self, config: ExperimentConfig, num_classes: int, device: str, run_id: str):
        self.config = config
        self.device = device
        self.num_features = TOTAL_FEATURES if config.features is None else len(config.features)
        self.device = device
        self.num_classes = num_classes
        self.run_id = run_id

    def init_model(self):
        self.model = ConvLiquidEEG(
            liquid_units=self.config.liquid_units,
            seq_length=self.config.sequence_length,
            num_classes=self.num_classes,
            eeg_channels=self.num_features, # TODO: Parametrize
            dropout=self.config.dropout,
        ).to(self.device)

        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(params=self.model.parameters(), lr=self.config.learning_rate)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', verbose=True, patience=1)

    def train_model(self, num_epochs: int, train_ds: IterableDataset, valid_ds: IterableDataset, test_ds: IterableDataset, model_id = None) -> tuple:

        # if os.path.exists('./temp'):
        #     temp_files = glob.glob('./temp/*')
        #     for temp_file in temp_files: os.remove(temp_file)
        # else:
        #     os.makedirs('./temp')         

        model_id = model_id if model_id else self.config.name

        # Dataloaders
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, num_workers=0, drop_last=True, shuffle=False)
        valid_loader = DataLoader(valid_ds, batch_size=self.config.batch_size, num_workers=0, drop_last=True, shuffle=False)

        experiment_history = []
        best_val_acc = 0.0
        for t in range(num_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss, train_acc = train_loop(train_loader, self.model, self.loss_fn, self.optimizer, self.device)
            val_loss, val_acc = val_loop(valid_loader, self.model, self.loss_fn, self.device)
            self.lr_scheduler.step(train_loss)
            epoch = {'val': {'loss': val_loss, 'acc': val_acc}, 'train': {'loss': train_loss, 'acc': train_acc}}
            print(epoch)
            experiment_history.append(epoch)
            # save the best model based on val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f'./temp/model_{model_id}.pt')

        # Test run
        test_loader = DataLoader(test_ds, batch_size=self.config.batch_size, num_workers=0, drop_last=True, shuffle=False)
        # load the best model to test on test set
        self.model.load_state_dict(torch.load(f'./temp/model_{model_id}.pt'))
        test_loss, test_acc = val_loop(test_loader, self.model, self.loss_fn, self.device)

        print("Done!")

        return test_loss, test_acc, experiment_history

    def start_k_fold_training(self, num_epochs: int):
        data_files = get_all_experiment_files()
        test_file = data_files.pop(random.randint(0, len(data_files) - 1))

        test_ds = CsvEEGIterableDataset(
            csv_files=[test_file],
            target_label=self.config.target_label,
            sequence_length=self.config.sequence_length,
            sequence_overlap=self.config.sequence_overlap,
            features=self.config.features,
        )
        k_folding_ds = KFoldSplitDataset(
            csv_files=data_files,
            target_label=self.config.target_label,
            sequence_length=self.config.sequence_length,
            sequence_overlap=self.config.sequence_overlap,
            features=self.config.features,
        )

        test_accs = []
        test_losses = []

        for idx in range(len(k_folding_ds)):
            print(f'>>> START FOLD TRAINING [{idx}/{len(k_folding_ds)}]\n')
            train_ds, valid_ds = k_folding_ds[idx]
            # Init model for each fold
            self.init_model()
            model_id = f'{self.config.name}_fold_{idx}'
            test_loss, test_acc, history = self.train_model(num_epochs, train_ds, valid_ds, test_ds, model_id=model_id)

            test_losses.append(test_loss)
            test_accs.append(test_acc)

        avg_loss = np.mean(test_losses)
        avg_acc = np.mean(test_accs)
        self.log_experiment(history, avg_loss, avg_acc)


    def log_experiment(self, experiment_history: list[dict], test_loss: float, test_acc: float):
        train_log = {
            'num_params': count_parameters(self.model),
            'decay_rate': self.config.decay_rate,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'units': self.config.liquid_units,
            'num_epochs': EPOCHS,
            'sequence_length': self.config.sequence_length,
            'sequence_overlap': self.config.sequence_overlap,
            'dropout': self.config.dropout,
            'features': self.config.features,
            'train_history': experiment_history,
            'test_loss': test_loss,
            'test_acc': test_acc,
        }
        experiment_log_path = f'{LOG_BASE_DIR}/{self.run_id}/{self.config.name}'

        os.makedirs(experiment_log_path, exist_ok=True)
        log_name = f'{experiment_log_path}/history_log.json'

        with open(log_name, 'w', encoding='utf-8') as f:
            json.dump(train_log, f, ensure_ascii=False, indent=4)

class ExperimentFramework:
    experiments: list[Experiment]
    epochs: int = EPOCHS

    def __init__(self):
        config_list: list[ExperimentConfig] = []

        with open('./experiments-config.json') as f:
            experimental_config = json.load(f)
            self.epochs = experimental_config.get('train_epochs', EPOCHS)
            config_list = [ExperimentConfig(**data) for data in experimental_config.get('experiments', [])]


        run_id = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.experiments = [Experiment(config=config, num_classes=NUM_CLASSES, device=DEVICE, run_id=run_id) for config in config_list]

    def start(self):
        for experiment in self.experiments:
            experiment.start_k_fold_training(EPOCHS)
