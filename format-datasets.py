import numpy as np
import os
import scipy

from pandas import DataFrame

DATASETS_PATH = './datasets'
DATASETS_TREATED_PATH = './datasets/csv'
REST_STATE = 91
EXPERIMENT_FINISH = 92
INITAL_RELAX = 99

def to_dataframe(matlab_file: str) -> DataFrame:
    mat = scipy.io.loadmat(matlab_file, struct_as_record=True)
    mat = {k:v for k, v in mat.items() if k[0] != '_'}
    # Data is in the "o" variable of the matlab file
    content = mat['o']
    headers = [header[0] for header in np.concatenate(content['chnames'][0][0]).ravel()]
    
    df = DataFrame(content['data'][0][0], columns=headers)
    # Add markers column
    df['Marker'] = content['marker'][0][0]

    return df

def split_experiments(df: DataFrame) -> list[DataFrame]:
    # Split the dataset into experiments, based on the EDA performed.

    # Removal of data prior to experiment start.
    start_idx = df.loc[df['Marker'] == INITAL_RELAX].last_valid_index()
    start_idx = 0 if start_idx is None else start_idx
    df = df.truncate(before=start_idx + 1)
    # Remove data after experiment finished.
    end_idx = df.loc[df['Marker'] == EXPERIMENT_FINISH].first_valid_index()
    end_idx = -1 if end_idx is None else end_idx
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

def main():
    for file in os.listdir(DATASETS_PATH):
        if file.endswith('.mat'):
            dataset_path: str = os.path.join(DATASETS_PATH, file)
            df: DataFrame = to_dataframe(dataset_path)
            experiments: list[DataFrame] = split_experiments(df)

            for idx, experiment in enumerate(experiments):
                file_name = os.path.splitext(file)[0] + f'_experiment_{idx}.csv'
                treated_path = os.path.join(DATASETS_TREATED_PATH, file_name)
                experiment.to_csv(treated_path)

if __name__ == '__main__':
    main()
