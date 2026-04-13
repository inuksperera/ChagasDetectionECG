import os
import glob
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm
import wfdb
import ast
from sklearn.metrics import roc_auc_score, roc_curve, roc_curve
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


def load_dataset(path, sampling_rate, no_of_samples, release=False):
    # load and convert annotation data
    #colab code
    Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id', nrows=no_of_samples)
    # Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
    # Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data_ptbxl(Y, sampling_rate, path, no_of_samples)

    return X, Y


def load_raw_data_ptbxl(df, sampling_rate, path, no_of_samples):
    print('=================8 Chagas======================================')
    print(f'Loading PTB-XL data at {sampling_rate}Hz from {path}...')
    # if sampling_rate == 100:
    #     if os.path.exists(path + 'raw100.npy'):
    #         data = np.load(path+'raw100.npy', allow_pickle=True)
    #     else:
    #         data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
    #         data = np.array([signal for signal, meta in data])
    #         pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)

    if sampling_rate == 500:
        output_path = path + '291_raw500.npy'
        if os.path.exists(output_path):
            print('Loading existing raw500.npy from ' + path)
            try:
                data = np.load(output_path, allow_pickle=True)
                print(f'Successfully loaded {len(data)} records.')
                return data
            except Exception as e:
                print(f"Error loading {output_path}: {e}")
                print("The file appears to be corrupted. Please delete it from Google Drive and run again.")
                raise e
        else:
            print('Creating new raw500.npy at: ' + path)
            n_total = len(df)
            if no_of_samples is not None:
                n_total = min(no_of_samples, len(df))
            
            # Pre-allocate memory (21799 records at 500Hz is ~5.2GB)
            # This handles all operations in RAM and prevents file corruption
            data = np.empty((n_total, 5000, 12), dtype=np.float32)
            
            batch_size = 500
            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                print(f"Processing records {start} to {end} / {n_total}")

                batch_filenames = df.filename_hr.iloc[start:end]
                for idx, f in enumerate(tqdm(batch_filenames, desc="Batch")):
                    # Load directly into pre-allocated array space
                    data[start + idx] = wfdb.rdsamp(path + f)[0]

            print(f"Saving combined dataset to {output_path}...")
            np.save(output_path, data)
            print('=================PTB-XL DATASET READY======================================')

    return data
