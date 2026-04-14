import numpy as np
import pandas as pd
from utils import return_purified, return_purified_feature, return_unique
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import os
import wfdb
from tqdm import tqdm
import glob
import h5py
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

# Function to downsample waves using resampling
def downsample_waves(waves, new_size):
    return np.array([resample(wave, new_size, axis=1) for wave in waves])


# Function to normalize the voltage range of each ECG signal per lead, scaling all signals to a consistent range. (used for combining PTB-XL and SaMi-Trop since they have different voltage ranges)
def normalize_ecg_per_lead(data):
    """Z-score normalize each lead of each ECG independently.
    data shape: (n_samples, n_leads, n_timesteps)"""
    eps = 1e-8
    mean = np.mean(data, axis=2, keepdims=True)
    std = np.std(data, axis=2, keepdims=True)
    return (data - mean) / (std + eps)


def remove_invalid_samples(waves, index=False):
    """
    Remove samples with NaN values or samples with the first 15 timesteps being all zeros.
    
    Args:
    waves (numpy.ndarray): The input array with shape (n_samples, n_channels, n_timesteps).
    index (bool): If True, output is indices of valid samples; otherwise, output is the valid samples themselves.
    """
    # Remove samples with NaN values
    nan_mask = np.isnan(waves).any(axis=(1, 2))
    
    # Remove samples with all zeros in the first 15 timesteps
    zero_mask = (np.abs(waves[:, :, :15]).sum(axis=(1, 2)) == 0)
    
    # Combine masks to find valid samples
    valid_indices = ~(nan_mask | zero_mask)
    
    # Print the number of invalid samples
    n_invalid = np.sum(~valid_indices)
    print(f'invalid samples: {n_invalid}')
    
    if index:
        return valid_indices
    else:
        return waves[valid_indices]
    

# Custom dataset class
class ECGDataset(Dataset):
    def __init__(self, waves, labels, transform=None):
        self.waves = torch.tensor(waves, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        wave = self.waves[idx]
        label = self.labels[idx]

        if self.transform:
            wave = self.transform(wave)
        return wave, label

# Custom dataset class
class ECGDataset_pretrain(Dataset):
    def __init__(self, waves):
        self.waves = torch.tensor(waves, dtype=torch.float32)

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        wave = self.waves[idx]
        return wave

def extract_diagnosis_code(record):
    for comment in record.comments:
        if comment.startswith('Dx:'):
            return comment.split(': ')[1]
    return None

def get_ecg_data(data_dir, reduced_lead=True, use_more=False, dx=False):
    """
    Read ECG data from the specified directory and return the data as a numpy array.

    Args:
    data_dir (str): The directory containing the ECG data files.
    reduced_lead (bool): If True, only eight leads are used: I, II, V1, V2, V3, V4, V5, V6.
                       : If False, all 12 leads are used.
    use_more (bool): If True, ECGs with more than 10s of data are split into multiple segments of 10s each.
    dx (bool): If True, extract the diagnosis code from the comments in the header file.

    Returns:
    np.ndarray: The ECG data array with shape (n_samples, n_channels, n_timesteps).
    np.ndarray (optional): Diagnostic code if dx is True.
    """
    ecg_records = []
    ecg_labels = []
    segment_length = 5000  # Length of each segment in samples (10 seconds at 500 Hz)

    for filename in os.listdir(data_dir):
        if filename.endswith('.hea'):
            record_name = os.path.splitext(filename)[0]
            print(f'Processing record: {record_name}')
            try:
                record = wfdb.rdrecord(os.path.join(data_dir, record_name))
            except Exception as e:
                print(f"Skipping broken record {record_name}: {e}")
                continue  # skip this broken record and keep going

            ecg_data = record.p_signal
            ecg_label = extract_diagnosis_code(record) if dx else None

            # Resample the data if the sampling frequency is not 500 Hz
            if record.fs != 500:
                # Calculate the new length for resampling
                new_length = int((500 / record.fs) * record.sig_len)
                ecg_data = resample(ecg_data, new_length)

            # Process the data in segments
            if ecg_data.shape[0] >= segment_length:
                ecg_records.append(ecg_data[:segment_length])
                ecg_labels.append(ecg_label)
    
    # Convert lists to numpy arrays
    ecg_records = np.stack(ecg_records, axis=0)
    ecg_records = ecg_records.transpose(0, 2, 1)  # (n_samples, n_channels, n_timesteps)
    ecg_labels = np.array(ecg_labels)

    if reduced_lead:
        # Keep only the leads I, II, V1, V2, V3, V4, V5, V6
        ecg_records = np.concatenate((ecg_records[:, :2, :], ecg_records[:, 6:, :]), axis=1)

    if dx:
        return ecg_records, ecg_labels
    else:
        return ecg_records


def subdirectory(data_dir):
    contents = os.listdir(data_dir)
    data_dirs = [d for d in contents if os.path.isdir(os.path.join(data_dir, d))]
    return data_dirs

def waves_cinc(data_dir, reduced_lead=True):
    waves = []
    for subdir in subdirectory(data_dir):
        for minibatch in subdirectory(os.path.join(data_dir, subdir)):
            ecg_data = get_ecg_data(os.path.join(data_dir, subdir, minibatch), reduced_lead=reduced_lead)
            waves.append(ecg_data)

    waves = np.concatenate(waves, axis=0)
    waves = remove_invalid_samples(waves)
    return waves

def waves_shao(data_dir, reduced_lead=True):
    waves = []
    for subdir in subdirectory(data_dir):
        for minibatch in subdirectory(os.path.join(data_dir, subdir)):
            ecg_data = get_ecg_data(os.path.join(data_dir, subdir, minibatch), reduced_lead=reduced_lead)
            waves.append(ecg_data)

    waves = np.concatenate(waves, axis=0)
    waves = remove_invalid_samples(waves)
    return waves

# def waves_shao(data_dir, reduced_lead=True):
#     waves = get_ecg_data(data_dir, reduced_lead=reduced_lead, dx=False)
#     waves = remove_invalid_samples(waves)
#     return waves

class Code15Dataset(Dataset):
    def __init__(self, data_dir, transform=None, reduced_lead=True, downsample=True, use_cache=True):
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(data_dir, '*.hdf5'))
        self.transform = transform
        self.reduced_lead = reduced_lead 
        self.downsample = downsample
        self.file_indices = []
        self._cache = {}

        # Cache file path
        self.cache_file = os.path.join(data_dir, 'file_indices_cache.npy')
         
        # Precompute the indices for each file and filter out padded waves
        self._compute_file_indices(use_cache)

    def _compute_file_indices(self, use_cache):
        if use_cache and os.path.exists(self.cache_file):
            self.file_indices = np.load(self.cache_file, allow_pickle=True).tolist()
        else:
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(self._process_file, enumerate(self.files)))
            for file_idx, indices in results:
                self.file_indices.extend([(file_idx, i) for i in indices])

            # Save the generated file indices to cache
            if use_cache:
                np.save(self.cache_file, np.array(self.file_indices, dtype=object))

    def _process_file(self, file_idx_and_name):
        file_idx, filename = file_idx_and_name
        valid_indices = []
        with h5py.File(filename, 'r') as f:
            num_samples = f['tracings'].shape[0]
            for i in range(num_samples):
                wave = np.array(f['tracings'][i])
                if not np.all(wave[:10] == 0):  # Check if first 10 timesteps are not all zeros
                    valid_indices.append(i)
        return file_idx, valid_indices

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        if idx >= len(self.file_indices):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self.file_indices)}")
        file_idx, sample_idx = self.file_indices[idx]
        filename = self.files[file_idx]

        # Check cache first
        if (file_idx, sample_idx) in self._cache:
            wave = self._cache[(file_idx, sample_idx)]
        else:
            with h5py.File(filename, 'r') as f:
                wave = np.array(f['tracings'][sample_idx])
            self._cache[(file_idx, sample_idx)] = wave  # Cache the loaded wave

        # Transpose the wave so channels come first
        wave = wave.T
        
        if self.reduced_lead:
            wave = wave[[0, 1, 6, 7, 8, 9, 10, 11], :]
        
        if self.downsample:
            wave = resample(wave, 2500, axis=1)

        if self.transform:
            wave = self.transform(wave)
            
        return torch.tensor(wave, dtype=torch.float)


def waves_samitrop(data_dir, task='multilabel', reduced_lead=False, downsample=True):
    from samitrop_utils import load_dataset, load_raw_data_samitrop
    # compute_label_aggregations, select_data

    sampling_frequency=500
    #colab code
    # no_of_samples = 291;
    no_of_samples = 1631;

    # Load Sami-Trop data
    #colab code
    data, raw_labels = load_dataset(data_dir, sampling_frequency, no_of_samples)
    # data = data.transpose(0,2,1)


    # === TEST PRINT BLOCK ===
    print("\n" + "="*30)
    print("DEBUG: SaMi-Trop Partial implementation Test")
    print(f"Data type: {type(data)}")
    if hasattr(data, 'shape'):
        print(f"Data shape: {data.shape} (Expected: ({no_of_samples}, 5000, 12))")
    
    print(f"Raw Labels type: {type(raw_labels)}")
    print("Raw Labels head:")
    print(raw_labels.head())  # Prints first 5 rows of the CSV
    
    # Check if 'Chagas' or relevant label columns exist
    # if 'is_chagas' in raw_labels.columns or 'label' in raw_labels.columns:
    #     print("\nLabel Column values (first 5):")
    #     # Adjust column name below based on your exams.csv structure
    #     col = 'is_chagas' if 'is_chagas' in raw_labels.columns else raw_labels.columns[0]
    #     print(raw_labels[col].head())
    # print("="*30 + "\n")
    # ========================


    # === DATA PROCESSING ===
    # Transpose to (n_samples, n_channels, n_timesteps)
    data = data.transpose(0, 2, 1)

    # Select leads if needed
    if reduced_lead:
        # Keep only leads I, II, V1, V2, V3, V4, V5, V6

        # print(f"\n--- Lead Reduction Debug (PTB-XL) ---")
        # print(f"Sample values before reduction (12 leads, first 5 timesteps):\n{data[0, :, :5]}")
        
        data = np.concatenate((data[:, :2, :], data[:, 6:, :]), axis=1)
        
        # print(f"Sample values after reduction (8 leads, first 5 timesteps):\n{data[0, :, :5]}")
        # print("PTBXL SHAPE AFTER REDUCED LEAD: " + str(data.shape))
        # print("------------------------------------\n")

    # Downsample if needed
    if downsample:
        data = resample(data, 2500, axis=2)

    # Since the entire SaMi-Trop dataset is Chagas Positive (hardcoded in prepare_samitrop_data.py), We just generate an array of 1s.
    if task == 'multilabel':
        # Create a completely positive binary matrix: shape (n_samples, 1)
        chagas_labels = np.ones((len(data), 1), dtype=np.float32)
    elif task == 'multiclass':
        # Create a completely positive 1D array: shape (n_samples,)
        # Assume class '1' represents Chagas positive
        chagas_labels = np.ones(len(data), dtype=np.int64)



    # print(raw_labels['age'].values)
    # print(raw_labels['age'].values.dtype)

    print("raw_labels.shape" + str(raw_labels.shape))  # (1631, 6)
    print('='*30)
    print('values before stratification')
    
    # Check if 551877 exists in the dataset before trying to print it
    target_ids = [551877, 158100, 709486, 253958, 587042, 204540, 446560, 96292, 181992]
    for target_id in target_ids:
        if target_id in raw_labels.index:
            # Get the positional integer index (0-based) for this specific exam_id
            target_idx = raw_labels.index.get_loc(target_id)
            
            print(f"Exam ID: {target_id} is at Positional Index: {target_idx}")
            print(f"Lead 0, first 3 timesteps for {target_id}: {data[target_idx, 0, :3]}")
        else:
            print(f"Warning: Exam ID {target_id} was not found in raw_labels.index")
            


    # Perform the 70-20-10 split (stratify by age + gender for balance)
    raw_labels['age_bin'] = pd.qcut(
        raw_labels['age'], 
        q=10, 
        labels=False, 
        duplicates='drop'    
    )

    # new column for stratification (combined age bins and gender)
    raw_labels['stratification'] = raw_labels['age_bin'].astype(str) + "_" + raw_labels['is_male'].astype(str)

    # split both the raw labels, ecg data and chagas labels at once so that they follow the same order when shuffled
    waves_train, waves_temp, raw_labels_train, raw_labels_temp, chagas_labels_train, chagas_labels_temp = train_test_split(
        data,                # Numpy array with ECG data
        raw_labels,          # Raw labels from CSV
        chagas_labels,       # Chagas labels (1,0)
        test_size=0.3,
        random_state=42,
        stratify=raw_labels['stratification']
    )

    waves_validation, waves_test, raw_labels_validation, raw_labels_test, chagas_labels_validation, chagas_labels_test = train_test_split(
        waves_temp,
        raw_labels_temp, 
        chagas_labels_temp,
        test_size=1/3,
        random_state=42,
        stratify=raw_labels_temp['stratification']
    )

    # Clean up helper columns
    for split_df in [raw_labels_train, raw_labels_validation, raw_labels_test]:
        split_df.drop(columns=['age_bin', 'stratification'], inplace=True, errors='ignore')

    # print("raw_labels.shape" + str(raw_labels.shape) + str(raw_labels.columns))
    # print("raw_labels_train.shape" + str(raw_labels_train.shape) + str(raw_labels_train.columns))
    # print("raw_labels_temp.shape" + str(raw_labels_temp.shape) + str(raw_labels_temp.columns))
    # print("raw_labels_validation.shape" + str(raw_labels_validation.shape) + str(raw_labels_validation.columns))
    # print("raw_labels_test.shape" + str(raw_labels_test.shape) + str(raw_labels_test.columns))
    print("\n=== SANITY CHECK: TRAINING SET ===")
    print("Raw Labels (first 3):")
    print(raw_labels_train.head(3))
    print("Waves Array (first 3 shapes):", waves_train[:3].shape)
    print(f"Waves Data Snippet (Lead 0, first 3 timesteps):\n Patient 0: {waves_train[0, 0, :3]}\n Patient 1: {waves_train[1, 0, :3]}\n Patient 2: {waves_train[2, 0, :3]}")
    print("Chagas Labels (first 3):", chagas_labels_train[:3].flatten())

    print("\n=== SANITY CHECK: VALIDATION SET ===")
    print("Raw Labels (first 3):")
    print(raw_labels_validation.head(3))
    print("Waves Array (first 3 shapes):", waves_validation[:3].shape)
    print(f"Waves Data Snippet (Lead 0, first 3 timesteps):\n Patient 0: {waves_validation[0, 0, :3]}\n Patient 1: {waves_validation[1, 0, :3]}\n Patient 2: {waves_validation[2, 0, :3]}")
    print("Chagas Labels (first 3):", chagas_labels_validation[:3].flatten())

    print("\n=== SANITY CHECK: TESTING SET ===")
    print("Raw Labels (first 3):")
    print(raw_labels_test.head(3))
    print("Waves Array (first 3 shapes):", waves_test[:3].shape)
    print(f"Waves Data Snippet (Lead 0, first 3 timesteps):\n Patient 0: {waves_test[0, 0, :3]}\n Patient 1: {waves_test[1, 0, :3]}\n Patient 2: {waves_test[2, 0, :3]}")
    print("Chagas Labels (first 3):", chagas_labels_test[:3].flatten())
    print("==================================\n")

    print(data.shape)

    print(f"Train: {len(raw_labels_train)} (~70%)")
    print(f"Val:   {len(raw_labels_validation)} (~20%)")
    print(f"Test:  {len(raw_labels_test)} (~10%)")

    # 3. Save the split indices (or just the exam_ids)
    # np.save("sami_trop_train_idx.npy", df_train.index.values)
    # np.save("sami_trop_val_idx.npy",   df_val.index.values)
    # np.save("sami_trop_test_idx.npy",  df_test.index.values)

    """
    raw_labels: The original Pandas DataFrame from the CSV.
    labels: A processed Pandas DataFrame that includes a strat_fold column (used to decide which samples go to Train vs. Test).
    Y: The final NumPy Array (0's and 1's) that is used as the target for model training.
    """

    # stratified data for training 
    # waves_train = data[raw_labels_train.index]
    # labels_train = Y[raw_labels_train.index]

    ## stratified data for validation
    # waves_validation = data_[labels.strat_fold == 10]
    # labels_validation = Y[labels.strat_fold == 10]

    ## stratified data for testing
    # waves_test = data_[labels.strat_fold == 10]
    # labels_test = Y[labels.strat_fold == 10]

    # if task == 'multiclass':
    #     waves_train, labels_train = convert_to_multiclass(waves_train, labels_train)
    #     waves_test, labels_test = convert_to_multiclass(waves_test, labels_test)

    # return waves_train, waves_test, labels_train, labels_test

    # change to train and validation only (exclude test)
    return waves_train, waves_validation, chagas_labels_train, chagas_labels_validation


# Reimplemented the waves_ptbxl() function for chagas detection
def waves_ptbxl_chagas(data_dir, task='multilabel', reduced_lead=True, downsample=True):
    from ptbxl_utils import load_dataset
    
    assert task in ['multilabel', 'multiclass']

    # cat = 'superdiagnostic'
    # categories = ['all', 'diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm']
    # assert cat in categories, f'Invalid category: {cat}, choose from {categories}'

    sampling_frequency=500
    # colab code
    # no_of_samples = 291;
    no_of_samples = 21799;

    # Load PTB-XL data
    #colab code
    data, raw_labels = load_dataset(data_dir, sampling_frequency, no_of_samples)
    data = data.transpose(0,2,1)
    print("PTBXL SHAPE AFTER TRANSPOSE: " + str(data.shape))
    
    if downsample:
        data = np.array([resample(data[i], 2500, axis=1) for i in range(len(data))])
    
    if reduced_lead:
        data = np.concatenate([data[:,:2], data[:,6:]], axis=1)


    # Since the entire PTB-XL dataset is Chagas Negative, We just generate an array of 0s.
    if task == 'multilabel':
        # Create a completely negative binary matrix: shape (n_samples, 1)
        chagas_labels = np.zeros((len(data), 1), dtype=np.float32)
    elif task == 'multiclass':
        # Create a completely negative 1D array: shape (n_samples,)
        # Assume class '0' represents Chagas negative
        chagas_labels = np.zeros(len(data), dtype=np.int64)



    # Preprocess label data
    # labels = compute_label_aggregations(raw_labels, data_dir, cat)
    # Select relevant data and convert to one-hot
    # data_, labels, Y, _ = select_data(data, labels, cat, min_samples=0)
    # data_, labels, Y = 


    print("DATA SHAPE: " + str(data.shape))
    print("CHAGAS LABELS SHAPE: " + str(chagas_labels.shape))
    print("RAW LABELS SHAPE: " + str(raw_labels.shape))

    # 1-8 for training 
    waves_train = data[raw_labels.strat_fold < 9]
    labels_train = chagas_labels[raw_labels.strat_fold < 9]

    # 9 for validation
    waves_validation = data[raw_labels.strat_fold == 9]
    labels_validation = chagas_labels[raw_labels.strat_fold == 9]

    # 10 for testing
    waves_test = data[raw_labels.strat_fold == 10]
    labels_test = chagas_labels[raw_labels.strat_fold == 10]

    print("WAVES TRAIN SHAPE: " + str(waves_train.shape))
    print("LABELS TRAIN SHAPE: " + str(labels_train.shape))
    print("WAVES VALIDATION SHAPE: " + str(waves_validation.shape))
    print("LABELS VALIDATION SHAPE: " + str(labels_validation.shape))
    print("WAVES TEST SHAPE: " + str(waves_test.shape))
    print("LABELS TEST SHAPE: " + str(labels_test.shape))

    # if task == 'multiclass':
    #     waves_train, labels_train = convert_to_multiclass(waves_train, labels_train)
    #     waves_test, labels_test = convert_to_multiclass(waves_test, labels_test)

    return waves_train, waves_validation, labels_train, labels_validation


# Return the combined data from PTB-XL and SAMI-TROP with waves and labels for training and testing
def waves_combined_data(data_dir_ptbxl, data_dir_samitrop, task='multilabel', reduced_lead=True, downsample=True):
    
    # Load PTB-XL data
    waves_train_ptbxl, waves_test_ptbxl, labels_train_ptbxl, labels_test_ptbxl = waves_ptbxl_chagas(data_dir_ptbxl, task, reduced_lead=reduced_lead)

    # Save a snippet of the raw PTB-XL data before normalization (first 5 ECGs, Lead 0, first 10 timesteps)
    ptbxl_before = waves_train_ptbxl[:5, 0, :10].copy()
    ptbxl_train_min_before, ptbxl_train_max_before = np.min(waves_train_ptbxl), np.max(waves_train_ptbxl)
    ptbxl_test_min_before, ptbxl_test_max_before = np.min(waves_test_ptbxl), np.max(waves_test_ptbxl)

    # normalize the ranges for voltage values for combining the two datasets, scaling all signals to a consistent range
    waves_train_ptbxl = normalize_ecg_per_lead(waves_train_ptbxl)
    waves_test_ptbxl = normalize_ecg_per_lead(waves_test_ptbxl)

    print("\n" + "="*80)
    print("DEBUG: PTB-XL Normalization Check")
    print(f"TRAIN Range BEFORE: Min={ptbxl_train_min_before:.4f}, Max={ptbxl_train_max_before:.4f}")
    print(f"TRAIN Range AFTER : Min={np.min(waves_train_ptbxl):.4f}, Max={np.max(waves_train_ptbxl):.4f}")
    print(f"TEST Range BEFORE : Min={ptbxl_test_min_before:.4f}, Max={ptbxl_test_max_before:.4f}")
    print(f"TEST Range AFTER  : Min={np.min(waves_test_ptbxl):.4f}, Max={np.max(waves_test_ptbxl):.4f}")
    print("-" * 80)
    for i in range(5):
        print(f"Patient {i} BEFORE: {ptbxl_before[i]}")
        print(f"Patient {i} AFTER : {waves_train_ptbxl[i, 0, :10]}\n")
    print("="*80 + "\n")


    print("PTB-XL TRAIN SHAPE: " + str(waves_train_ptbxl.shape))
    print("PTB-XL TEST SHAPE: " + str(waves_test_ptbxl.shape))
    print("PTB-XL LABELS TRAIN SHAPE: " + str(labels_train_ptbxl.shape))
    print("PTB-XL LABELS TEST SHAPE: " + str(labels_test_ptbxl.shape))
   
    # Load SAMI-TROP data
    waves_train_samitrop, waves_test_samitrop, labels_train_samitrop, labels_test_samitrop = waves_samitrop(data_dir_samitrop, task, reduced_lead=reduced_lead)

    # Save a snippet of the raw SaMi-Trop data before normalization
    samitrop_before = waves_train_samitrop[:5, 0, :10].copy()
    sami_train_min_before, sami_train_max_before = np.min(waves_train_samitrop), np.max(waves_train_samitrop)
    sami_test_min_before, sami_test_max_before = np.min(waves_test_samitrop), np.max(waves_test_samitrop)

    # normalize the ranges for voltage values for combining the two datasets, scaling all signals to a consistent range
    waves_train_samitrop = normalize_ecg_per_lead(waves_train_samitrop)
    waves_test_samitrop = normalize_ecg_per_lead(waves_test_samitrop)

    print("\n" + "="*80)
    print("DEBUG: SAMI-TROP Normalization Check")
    print(f"TRAIN Range BEFORE: Min={sami_train_min_before:.4f}, Max={sami_train_max_before:.4f}")
    print(f"TRAIN Range AFTER : Min={np.min(waves_train_samitrop):.4f}, Max={np.max(waves_train_samitrop):.4f}")
    print(f"TEST Range BEFORE : Min={sami_test_min_before:.4f}, Max={sami_test_max_before:.4f}")
    print(f"TEST Range AFTER  : Min={np.min(waves_test_samitrop):.4f}, Max={np.max(waves_test_samitrop):.4f}")
    print("-" * 80)
    for i in range(5):
        print(f"Patient {i} BEFORE: {samitrop_before[i]}")
        print(f"Patient {i} AFTER : {waves_train_samitrop[i, 0, :10]}\n")
    print("="*80 + "\n")

    print("SAMI-TROP TRAIN SHAPE: " + str(waves_train_samitrop.shape))
    print("SAMI-TROP TEST SHAPE: " + str(waves_test_samitrop.shape))
    print("SAMI-TROP LABELS TRAIN SHAPE: " + str(labels_train_samitrop.shape))
    print("SAMI-TROP LABELS TEST SHAPE: " + str(labels_test_samitrop.shape))



    print("\n" + "="*150)
    print("DEBUG: DATASET VOLTAGE SCALE COMPARISON")
    print("-" * 150)
    # Average across patients (axis 0) and time (axis 2) to get per-lead mean
    ptbxl_mean = np.mean(waves_train_ptbxl, axis=(0, 2))
    sami_mean = np.mean(waves_train_samitrop, axis=(0, 2))
    ptbxl_abs_mean = np.mean(np.abs(waves_train_ptbxl), axis=(0, 2))
    sami_abs_mean = np.mean(np.abs(waves_train_samitrop), axis=(0, 2))

    print(f"PTB-XL Mean Voltage per Lead:     {ptbxl_mean}")
    print(f"SAMI-TROP Mean Voltage per Lead:  {sami_mean}")
    print(f"PTB-XL Mean |Voltage| per Lead:   {ptbxl_abs_mean}")
    print(f"SAMI-TROP Mean |Voltage| per Lead: {sami_abs_mean}")
    print("="*50 + "\n")



    # Concatenate the data
    # Training data
    waves_train_combined = np.concatenate((waves_train_ptbxl, waves_train_samitrop), axis=0)
    labels_train_combined = np.concatenate((labels_train_ptbxl, labels_train_samitrop), axis=0)

    # Testing data
    waves_test_combined = np.concatenate((waves_test_ptbxl, waves_test_samitrop), axis=0)
    labels_test_combined = np.concatenate((labels_test_ptbxl, labels_test_samitrop), axis=0)

    print("WAVES TRAIN COMBINED SHAPE: " + str(waves_train_combined.shape))
    print("LABELS TRAIN COMBINED SHAPE: " + str(labels_train_combined.shape))
    print("WAVES TEST COMBINED SHAPE: " + str(waves_test_combined.shape))
    print("LABELS TEST COMBINED SHAPE: " + str(labels_test_combined.shape))
    
# waves_train, waves_test, labels_train, labels_test = waves_samitrop(data_dir, task, reduced_lead=reduced_lead)
    
#     elif dataset == 'ptbxl_chagas':
#         waves_train, waves_test, labels_train, labels_test = waves_ptbxl_chagas(data_dir, task, reduced_lead=reduced_lead)

        
    return waves_train_combined, waves_test_combined, labels_train_combined, labels_test_combined


def waves_ptbxl(data_dir, task='multilabel', reduced_lead=True, downsample=True):
    from ptbxl_utils import load_dataset, compute_label_aggregations, select_data
    assert task in ['multilabel', 'multiclass']

    cat = 'superdiagnostic'
    categories = ['all', 'diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm']
    assert cat in categories, f'Invalid category: {cat}, choose from {categories}'

    sampling_frequency=500
    # colab code
    # no_of_samples = 291;
    no_of_samples = 21799;

    # Load PTB-XL data
    #colab code
    data, raw_labels = load_dataset(data_dir, sampling_frequency, no_of_samples)
    data = data.transpose(0,2,1)
    print("PTBXL SHAPE AFTER TRANSPOSE: " + str(data.shape))
    
    if downsample:
        data = np.array([resample(data[i], 2500, axis=1) for i in range(len(data))])
    
    if reduced_lead:
        data = np.concatenate([data[:,:2], data[:,6:]], axis=1)

    # Preprocess label data
    labels = compute_label_aggregations(raw_labels, data_dir, cat)
    # Select relevant data and convert to one-hot
    data_, labels, Y, _ = select_data(data, labels, cat, min_samples=0)

    # 1-9 for training 
    waves_train = data_[labels.strat_fold < 10]
    labels_train = Y[labels.strat_fold < 10]

    # 10 for validation
    waves_test = data_[labels.strat_fold == 10]
    labels_test = Y[labels.strat_fold == 10]

    if task == 'multiclass':
        waves_train, labels_train = convert_to_multiclass(waves_train, labels_train)
        waves_test, labels_test = convert_to_multiclass(waves_test, labels_test)

    return waves_train, waves_test, labels_train, labels_test


def waves_cpsc(data_dir, task='multilabel', reduced_lead=True, downsample=True):
    waves_cpsc = []
    labels_cpsc = []
    minibatches = []

    for minibatch in subdirectory(data_dir):
        ecg_data, ecg_labels = get_ecg_data(os.path.join(data_dir, minibatch), reduced_lead=True,  dx=True) #reduced_lead changed to True
        waves_cpsc.append(ecg_data)
        labels_cpsc.append(ecg_labels)
        minibatches.extend([minibatch] * len(ecg_data))

    waves_cpsc = np.concatenate(waves_cpsc, axis=0)
    labels_cpsc = np.concatenate(labels_cpsc, axis=0)
    minibatches = np.array(minibatches)

    # Remove samples with NaN values
    valid_indices = remove_invalid_samples(waves_cpsc, index=True)
    
    # remove samples with empty labels
    for i in range(len(labels_cpsc)):
        if labels_cpsc[i] == '':
            valid_indices[i] = False

    waves_cpsc = waves_cpsc[valid_indices]
    labels_cpsc = labels_cpsc[valid_indices]
    minibatches = minibatches[valid_indices]

    if downsample:
        waves_cpsc = downsample_waves(waves_cpsc, 2500)

    # Extract unique labels
    unique_labels = set()
    for label_str in np.unique(labels_cpsc):
        labels = label_str.split(',')
        unique_labels.update(labels)

    unique_labels = sorted(unique_labels)
    # Create a mapping from label to index
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    # Initialize the binary matrix
    labels_matrix = np.zeros((len(labels_cpsc), len(unique_labels)), dtype=int)

    # Populate the binary matrix
    for i, label_str in enumerate(labels_cpsc):
        labels = label_str.split(',')
        for label in labels:
            labels_matrix[i, label_to_index[label]] = 1

    labels_cpsc = labels_matrix

    test_indices = (minibatches == 'g7')
    train_indices = ~test_indices

    waves_train = waves_cpsc[train_indices]
    labels_train = labels_cpsc[train_indices]
    waves_test = waves_cpsc[test_indices]
    labels_test = labels_cpsc[test_indices]

    if task == 'multiclass':
        waves_train, labels_train = convert_to_multiclass(waves_train, labels_train)
        waves_test, labels_test = convert_to_multiclass(waves_test, labels_test)

    return waves_train, waves_test, labels_train, labels_test

def convert_to_multiclass(waves, labels):
    '''
    convert multi-label to multi-class by restricting to samples with only one label
    '''

    label_sums = np.sum(labels, axis=1)
    indices_with_one_label = np.where(label_sums == 1)[0]

    waves = waves[indices_with_one_label]
    labels = labels[indices_with_one_label]

    # ont-hot to integer
    labels = np.argmax(labels, axis=1)

    return waves, labels

def waves_from_config(config, reduced_lead=True): #reduced_lead changed to True
    # model_name = config['model_name']
    data_dir = config['data_dir']
    dataset = config['dataset']
    task = config['task']

    # if model_name == 'st_mem':
    #     reduced_lead = False

    if dataset == 'ptbxl':
        waves_train, waves_test, labels_train, labels_test = waves_ptbxl(data_dir, task, reduced_lead=reduced_lead)

    elif dataset == 'cpsc':
        waves_train, waves_test, labels_train, labels_test = waves_cpsc(data_dir, task, reduced_lead=reduced_lead)
        
    elif dataset == 'samitrop':
        waves_train, waves_test, labels_train, labels_test = waves_samitrop(data_dir, task, reduced_lead=reduced_lead)
    
    elif dataset == 'ptbxl_chagas':
        waves_train, waves_test, labels_train, labels_test = waves_ptbxl_chagas(data_dir, task, reduced_lead=reduced_lead)

    elif dataset == 'combined_data':
        data_dir_ptbxl = config['data_dir_ptbxl']
        data_dir_samitrop = config['data_dir_samitrop']

        waves_train, waves_test, labels_train, labels_test = waves_combined_data(data_dir_ptbxl, data_dir_samitrop, task, reduced_lead=reduced_lead)

    # # st_mem needs shorter waves 
    # if model_name == 'st_mem':
    #     waves_train = waves_train[:, :, 125:-125]
    #     waves_test = waves_test[:, :, 125:-125]

    return waves_train, waves_test, labels_train, labels_test




# Sami-Trop dataset Loading
