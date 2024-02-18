import glob
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from pydantic import BaseModel
from torcheeg.model_selection import train_test_split_per_subject_cross_trial, train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from constants import DEVICE

from dataset import BCI_C_IV_2A_NUM_CLASSES, DEFAULT_BANDS, TOTAL_FEATURES, CsvEEGIterableDataset, KFoldSplitDataset, get_all_experiment_files, get_bci_competition_dataset
from model import ModelType, count_parameters, get_model_instance
from training import EPOCHS, val_loop, train_loop

DEFAULT_EXPERIMENTS = './experiments-config.json'
LOG_BASE_DIR = './log/experiments'

class ExperimentConfig(BaseModel):
    name: str
    target_label: str
    learning_rate: float
    liquid_units: int
    dropout: float
    batch_size: int
    sequence_length: int
    dt: int
    sequence_overlap: float
    features: list[int] | None = None


class Experiment:
    def __init__(self, model_type: ModelType, config: ExperimentConfig, num_classes: int, device: str, run_id: str):
        self.model_type = model_type
        self.config = config
        self.device = device
        self.num_features = TOTAL_FEATURES if config.features is None else len(config.features)
        self.device = device
        self.num_classes = num_classes
        self.run_id = run_id

    def init_model(self):
        self.model = get_model_instance(self.model_type, BCI_C_IV_2A_NUM_CLASSES, **dict(self.config))

        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(params=self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=4, min_lr=1e-5, threshold=1e-3)

    def train_model(self, num_epochs: int, train_ds, valid_ds, test_ds = None, model_id = None) -> tuple:
        self.num_epochs = num_epochs
        if os.path.exists('./temp'):
            temp_files = glob.glob('./temp/*')
            for temp_file in temp_files: os.remove(temp_file)
        else:
            os.makedirs('./temp', exist_ok=True)         

        model_id = model_id if model_id else self.config.name

        # Dataloaders
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, num_workers=4, drop_last=True, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.config.batch_size, num_workers=4, drop_last=True, shuffle=False)

        experiment_history = []
        best_val_acc = 0.0
        best_val_loss = 0.0
        best_epoch = None
        for t in range(num_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            epoch_time_start = time.time()
            train_loss, train_acc = train_loop(train_loader, self.model, self.loss_fn, self.optimizer, self.device)
            val_loss, val_acc = val_loop(valid_loader, self.model, self.loss_fn, self.device)
            epoch_time = time.time() - epoch_time_start
            self.lr_scheduler.step(train_loss)
            print(f'>>>>>>>>>>>>>>>>>>> LEARNING RATE: {self.lr_scheduler.get_last_lr()}')
            epoch = {'time': epoch_time, 'val': {'loss': val_loss, 'acc': val_acc}, 'train': {'loss': train_loss, 'acc': train_acc}}
            print(epoch)
            experiment_history.append(epoch)
            # save the best model based on val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = t
                torch.save(self.model.state_dict(), f'./temp/model_{model_id}.pt')

        if test_ds is None:
            model_loss = best_val_loss
            model_acc = best_val_acc
        else:
            # Test run
            test_loader = DataLoader(test_ds, batch_size=self.config.batch_size, num_workers=0, drop_last=True, shuffle=False)
            # load the best model to test on test set
            self.model.load_state_dict(torch.load(f'./temp/model_{model_id}.pt'))
            model_loss, model_acc = val_loop(test_loader, self.model, self.loss_fn, self.device)

        print("Done!")

        return best_epoch, model_loss, model_acc, experiment_history

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

    def train_bci_model(self, num_epochs: int, subject: str | None):
        eeg_bands = {'delta_gamma': [1, 40]} if self.model_type == ModelType.ONLY_LIQUID else DEFAULT_BANDS
        dataset = get_bci_competition_dataset(self.config.sequence_length, self.config.dt, eeg_bands, self.config.features)

        if subject is None:
            train_ds, val_ds = train_test_split(dataset, shuffle=True)
        else:
            train_ds, val_ds = train_test_split_per_subject_cross_trial(dataset, test_size=0.2, subject=subject, shuffle=True)
        
        self.init_model()

        best_epoch, model_loss, model_acc, history = self.train_model(num_epochs, train_ds, val_ds)
        print(f'>>>>>>>>>> FINISH EXPERIMENT')
        self.log_experiment(history, model_loss, model_acc, best_epoch, subject=subject)

    def log_experiment(self, experiment_history: list[dict], test_loss: float, test_acc: float, best_epoch: int, subject: str | None = None):
        model_label: str = ModelType.label(self.model_type)
        train_log = {
            'config_name': self.config.name,
            'experiment_type': 'INTERSUBJECT' if subject is None else f'INTRASUBJECT {subject}',
            'model_type': model_label,
            'num_params': count_parameters(self.model),
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'units': self.config.liquid_units,
            'num_epochs': self.num_epochs,
            'sequence_length': self.config.sequence_length,
            'sequence_overlap': self.config.sequence_overlap,
            'dt': self.config.dt,
            'dropout': self.config.dropout,
            'features': self.config.features,
            'train_history': experiment_history,
            'best_epoch': best_epoch,
            'test_loss': test_loss,
            'test_acc': test_acc,
        }
        experiment_log_path = f'{LOG_BASE_DIR}/{self.run_id}/{model_label}'

        os.makedirs(experiment_log_path, exist_ok=True)
        log_name = f'{experiment_log_path}/{self.config.name}.json'

        with open(log_name, 'w', encoding='utf-8') as f:
            json.dump(train_log, f, ensure_ascii=False, indent=4)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ExperimentFramework:
    experiments: list[Experiment]
    epochs: int = EPOCHS

    def __init__(self, model_type: ModelType | None = None, experiments_path: str = DEFAULT_EXPERIMENTS):
        config_list: list[ExperimentConfig] = []
        with open(experiments_path) as f:
            experimental_config = json.load(f)
            self.epochs = experimental_config.get('train_epochs', EPOCHS)
            config_list = [ExperimentConfig(**data) for data in experimental_config.get('experiments', [])]

        run_id = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        models = list(ModelType) if model_type is None else [model_type]
        self.experiments = []
        for model in models:
            self.experiments += [Experiment(model, config=config, num_classes=BCI_C_IV_2A_NUM_CLASSES, device=DEVICE, run_id=run_id) for config in config_list]

    def start(self,  subject: str | None = None):
        for experiment in self.experiments:
            seed_everything(42)
            experiment.train_bci_model(self.epochs, subject=subject)