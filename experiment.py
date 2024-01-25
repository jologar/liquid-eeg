import datetime
import json

from pydantic import BaseModel
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from constants import DEVICE

from dataset import TOTAL_FEATURES, EEGDataset, EEGIterableDataset
from model import ConvLiquidEEG, count_parameters
from training import EPOCHS, test_loop, train_loop

NUM_CLASSES = 4


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
    def __init__(self, config: ExperimentConfig, num_classes: int, device: str):
        self.config = config
        self.device = device
        num_features = TOTAL_FEATURES if config.features is None else len(config.features)

        self.loss_fn = CrossEntropyLoss()
        self.model = ConvLiquidEEG(
            liquid_units=config.liquid_units,
            seq_length=config.sequence_length,
            num_classes=num_classes,
            eeg_channels=num_features, # TODO: Parametrize
            dropout=config.dropout,
        ).to(device)
        self.optimizer = Adam(params=self.model.parameters(), lr=config.learning_rate)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', verbose=True, patience=1)

    def start_training(self, num_epochs: int, train_ds: EEGIterableDataset, valid_ds: EEGDataset):
        # Datasets
        # Dataloaders
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, num_workers=0, drop_last=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.config.batch_size, num_workers=0, drop_last=True)

        experiment_history = []
        for t in range(num_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss, train_acc = train_loop(train_loader, self.model, self.loss_fn, self.optimizer, self.device)
            test_loss, test_acc = test_loop(valid_loader, self.model, self.loss_fn, self.device)
            self.lr_scheduler.step(train_loss)
            epoch = {'val': {'loss': test_loss, 'acc': test_acc}, 'train': {'loss': train_loss, 'acc': train_acc}}
            print(epoch)
            experiment_history.append(epoch)
            train_loader.dataset.init_iterator()
            valid_loader.dataset.init_iterator()
        self.log_experiment(experiment_history=experiment_history)
        print("Done!")

    def log_experiment(self, experiment_history: list[dict]):
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
        }

        log_name = f'././log/framework/history_{self.config.name}_{datetime.datetime.timestamp(datetime.datetime.now())}.json'

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



        self.experiments = [Experiment(config=config, num_classes=NUM_CLASSES, device=DEVICE) for config in config_list]

    def start(self):
        for experiment in self.experiments:
            # Load data
            print('LOADING DATA')
            total_train = None
            with open(TRAIN_DS) as f:
                total_train = sum(1 for _ in f)

            ds = EEGIterableDataset(
                target_label=experiment.config.target_label,
                ds_path=TRAIN_DS,
                sequence_length=experiment.config.sequence_length,
                sequence_overlap=experiment.config.sequence_overlap,
                features=experiment.config.features,
                total_samples=total_train
            )

            ds_valid = EEGIterableDataset(
                target_label=experiment.config.target_label,
                ds_path=VALID_DS,
                sequence_length=experiment.config.sequence_length,
                sequence_overlap=experiment.config.sequence_overlap,
                features=experiment.config.features,
            )
            
            print('DATA LOADED')
            experiment.start_training(num_epochs=self.epochs, train_ds=ds, valid_ds=ds_valid)
