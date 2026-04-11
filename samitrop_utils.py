import os
import ast
import wfdb
import pandas as pd
import numpy as np
from tqdm import tqdm
import wfdb

#colab code
def load_dataset(path, sampling_rate, no_of_samples):
    # load and convert annotation data
    #colab code
    Y = pd.read_csv(os.path.join(path, 'exams.csv'), index_col='exam_id', nrows=no_of_samples)
    # Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
    # Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data_samitrop(Y, sampling_rate, path, no_of_samples)

    return X, Y


def load_raw_data_samitrop(df, sampling_rate, path, no_of_samples):
    print('=================4======================================')
    print(f'Loading SAMITROP data at {sampling_rate}Hz from {path}...')
    """
    if sampling_rate == 100:
        if os.path.exists(path + 'samitrop_output/samitrop_raw100.npy'):
            data = np.load(path+'samitrop_output/samitrop_raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'samitrop_raw100.npy', 'wb'), protocol=4)
    """
    
    if sampling_rate == 500:
        output_path = path + 'samitrop_output/samitrop_raw500.npy'
        if os.path.exists(output_path):
            print('Loading existing samitrop_raw500.npy from ' + output_path)
            data = np.load(output_path, allow_pickle=True)
            if no_of_samples is not None:
                data = data[:no_of_samples]
            print('Successfully loaded existing samitrop_raw500.npy from ' + output_path)
        else:
            print('Loading dataset in path: ' + path + "samitrop_output/")
            batch_size = 100          # tune: 1000–4000 depending on session
            n_total = len(df)
            if no_of_samples is not None:
                n_total = no_of_samples
            first_batch = True

            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                print(f"Processing records {start+1} to {end} / {n_total}")

                #######################################################################
                # batch_filenames = df.filename_hr.iloc[start:end]
                # batch_data = [wfdb.rdsamp(path + f)[0] for f in tqdm(batch_filenames, desc="Batch")]
                
                # could do path + exam_id here
                batch_filenames = df.index[start:end]
                batch_data = [wfdb.rdsamp(path + "samitrop_output/" +  str(f))[0] for f in tqdm(batch_filenames, desc="Batch")]
                #wfdb.rdsamp(path + "samitrop_output/" +  f )[0]
                batch_array = np.array(batch_data, dtype=np.float32)   # (batch_size, 5000, 12)

                if first_batch:
                    np.save(output_path, batch_array)
                    first_batch = False
                else:
                    # Append by loading + concat + overwrite
                    existing = np.load(output_path)
                    combined = np.concatenate([existing, batch_array], axis=0)
                    np.save(output_path, combined)
                    del existing, combined   # free memory 

                print(f"Saved")
                del batch_data, batch_array
               
            print(f"loop ended")
            print(f'Loading {output_path}.py')
            data = np.load(output_path, allow_pickle=True)
            print(f"Saved & loaded {output_path} ({os.path.getsize(output_path)/1e9:.2f} GB)")
    return data