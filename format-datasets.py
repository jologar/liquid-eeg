import numpy as np
import os
import glob
import pandas as pd
import scipy

from random import shuffle

from imblearn.under_sampling import RandomUnderSampler
from pandas import DataFrame

VALIDATION_FILE = 'validation-eeg-data.csv'
TRAIN_FILE = 'train-eeg-data.csv'
DATASETS_PATH = './datasets'
DATASETS_TREATED_PATH = './datasets/csv'
RHO_THRESHOLD = 10
REST_STATE = 91
EXPERIMENT_FINISH = 92
INITAL_RELAX = 99
INVALID_STATE = 90
SPLIT_RATIO = 0.8


def to_dataframe(matlab_file: str) -> DataFrame:
    mat = scipy.io.loadmat(matlab_file, struct_as_record=True)
    mat = {k:v for k, v in mat.items() if k[0] != '_'}
    # Data is in the "o" variable of the matlab file
    content = mat['o']
    headers = [header[0] for header in np.concatenate(content['chnames'][0][0]).ravel()]
    
    df = DataFrame(content['data'][0][0], columns=headers)
    # Add markers column
    df['Marker'] = content['marker'][0][0]

    # Delete invalid marker states
    df = df.drop(df.loc[df['Marker'] == INVALID_STATE].index)
    df = df.reset_index(drop=True)

    return df


def split_experiments(df: DataFrame) -> list[DataFrame]:
    # Split the dataset into experiments, based on the EDA performed.

    # Removal of data prior to experiment start.
    start_idx = df.loc[df['Marker'] == INITAL_RELAX].last_valid_index()
    start_idx = 0 if start_idx is None else start_idx + 1
    # Remove data after experiment finished.
    end_idx = df.loc[df['Marker'] == EXPERIMENT_FINISH].first_valid_index()
    if end_idx is None:
        df = df[:-1]
    else:
        df = df.truncate(after=end_idx - 1)

    experiments = []
    while start_idx is not None:
        end_of_experiment_idx = df[start_idx:].loc[df['Marker'] == REST_STATE].first_valid_index()
        if end_of_experiment_idx:
            experiment = df[start_idx:end_of_experiment_idx]
            experiments.append(experiment)
            start_idx = df[end_of_experiment_idx:].loc[df['Marker'] != REST_STATE].first_valid_index()
        else:
            experiment = df[start_idx:]
            experiments.append(experiment)
            start_idx = None

    return experiments


def calculate_rho(df: DataFrame) -> float:
    class_counts = df['Marker'].value_counts().sort_index()
    return class_counts[np.argmax(class_counts)] / class_counts[np.argmin(class_counts)] 


def balance_dataset(df: DataFrame) -> DataFrame:
    rho = calculate_rho(df)
    if rho > RHO_THRESHOLD:
        # Per EDA the imbalance comes from class 0
        majority_samples = len(df.loc[df['Marker'] == 0])
        majority_ratio = majority_samples/(len(df)*len(np.unique(df['Marker'])))
        # The samples to keep from the majority class
        sample = df.loc[df['Marker'] == 0].sample(frac=majority_ratio)
        # Remove all imbalanced class occurrences
        df = df.loc[df['Marker'] != 0]
        # Add the undersampled class
        df = sample.combine_first(df)

    return df


def random_balance_dataset(df: DataFrame) -> DataFrame:
    rho = calculate_rho(df)
    if rho > RHO_THRESHOLD:
        rus = RandomUnderSampler(random_state=0)
        y = df['Marker']
        X = df.loc[:, df.columns != 'Marker']

        df_rus, y_rus = rus.fit_resample(X, y)
        df_rus['Marker'] = y_rus
        df = df_rus

    return df


def store_in_big_file(big_file_path: str, experiment: DataFrame):
    # Append to one big csv dataset
    if os.path.isfile(big_file_path):
        experiment.to_csv(big_file_path, mode='a', index=False, header=False)
    else:
        experiment = experiment.reset_index(drop=True)
        experiment.to_csv(big_file_path, index=False)


def main():
    total_experiments: list[DataFrame] = []
    for file in os.listdir(DATASETS_PATH):
        print(file)
        if file.endswith('.mat'):
            dataset_path: str = os.path.join(DATASETS_PATH, file)
            df: DataFrame = to_dataframe(dataset_path)
            experiments: list[DataFrame] = split_experiments(df)
                
            for idx, experiment in enumerate(experiments):
                experiment = balance_dataset(experiment)
                # experiment.drop(experiment[experiment['Marker'] == 0].index, inplace=True)
                total_experiments.append(experiment)

                # Store experiment in specific file            
                file_name = os.path.splitext(file)[0] + f'_experiment_{idx}.csv'
                treated_path = os.path.join(DATASETS_TREATED_PATH, file_name)
                
                experiment.to_csv(treated_path)
    shuffle(total_experiments)
    split_idx = int(len(total_experiments) * SPLIT_RATIO)
    
    for idx, experiment in enumerate(total_experiments):
        file_path = f'{DATASETS_TREATED_PATH}/{TRAIN_FILE}' if idx <= split_idx else f'{DATASETS_TREATED_PATH}/{VALIDATION_FILE}'
        store_in_big_file(file_path, experiment)
      

def raw():
    files = os.listdir(DATASETS_PATH)
    shuffle(files)
    split_idx = int(len(files) * SPLIT_RATIO)
    
    for idx, file in enumerate(files):
        if file.endswith('.mat'):
            dataset_path: str = os.path.join(DATASETS_PATH, file)
            df: DataFrame = to_dataframe(dataset_path)

            dest_file_name = './datasets/csv/raw-train-eeg-data.csv' if idx <= split_idx else './datasets/csv/raw-validation-eeg-data.csv'

            # Append to one big csv dataset
            if os.path.isfile(dest_file_name):
                df.to_csv(dest_file_name, mode='a', index=False, header=False)
            else:
                df = df.reset_index(drop=True)
                df.to_csv(dest_file_name, index=False)

if __name__ == '__main__':
    main()
