import numpy as np
from detect_disease import detect_disease
import torch
import os
import wfdb
from ecg_data import normalize_ecg_per_lead
from scipy.signal import resample

# ==============================================================================
# CHAGAS DISEASE DETECTION INFERENCE SCRIPT
# ==============================================================================

# 1. Prepare ECG Input Data
# The model expects a shape of [Batch, Leads, Time] or [Leads, Time]
# Required: 8 leads (I, II, V1, V2, V3, V4, V5, V6), 2500 sampling points (5 seconds @ 500Hz)
# Leads order: I, II, V1, V2, V3, V4, V5, V6

# Load your actual data here:

dirname = os.path.dirname(__file__)


# Load real ECG data from WFDB
record_path = os.path.join(dirname, 'testing data/positive/20122')
if not (os.path.exists(record_path + '.dat') and os.path.exists(record_path + '.hea')):
    print(f"ECG record files not found: {record_path}.dat/.hea")
    exit()

record = wfdb.rdrecord(record_path)
print("Lead names in WFDB record:", record.sig_name)
all_leads = record.p_signal.T  # shape: (num_leads, N)

# Print a sample of the first 10 values for each lead
for idx, name in enumerate(record.sig_name):
    print(f"Lead {idx} ({name}): {all_leads[idx, :10]}")

# Select leads: I, II, V1, V2, V3, V4, V5, V6 (indices may vary by dataset, but usually 0,1,6,7,8,9,10,11)
lead_indices = [0, 1, 6, 7, 8, 9, 10, 11]
print("Selected lead names:", [record.sig_name[i] for i in lead_indices])
selected_leads = all_leads[lead_indices, :]

# Print a sample of the selected leads
for i, idx in enumerate(lead_indices):
    print(f"Selected Lead {i} ({record.sig_name[idx]}): {selected_leads[i, :10]}")

# Resample to 2500 points per lead
num_points = 2500
from scipy.signal import resample
selected_leads = resample(selected_leads, num_points, axis=1)

# Add batch dimension: (1, 8, 2500)
ecg_input = selected_leads[np.newaxis, ...].astype(np.float32)


# Print min, max, mean for sanity check
print("Selected leads stats before normalization:")
print("Min:", np.min(ecg_input), "Max:", np.max(ecg_input), "Mean:", np.mean(ecg_input))


# Try different scaling factors, always use threshold=0.5
scales = [1, 10, 20]
for scale in scales:
    print(f"\n=== Trying input divided by {scale} before normalization ===")
    scaled_input = ecg_input / scale
    print("Stats before normalization: Min:", np.min(scaled_input), "Max:", np.max(scaled_input), "Mean:", np.mean(scaled_input))
    norm_input = normalize_ecg_per_lead(scaled_input)
    print("Stats after normalization: Min:", np.min(norm_input), "Max:", np.max(norm_input), "Mean:", np.mean(norm_input))

    # Run detection with threshold=0.5
    print(f"\nRunning detection for scale 1/{scale} (threshold=0.5)...")
    try:
        result = detect_disease(
            ecg_input=norm_input,
            combined_ckpt_path=combined_ckpt_path, 
            num_classes=1,  # 1 for binary classification (Chagas Positive/Negative)
            device='cuda' if torch.cuda.is_available() else 'cpu',
            threshold=0.5
        )
        prob_val = result['probability']
        pred_val = result['prediction']
        percent = prob_val.flatten() * 100
        print(f"Probability (%): {percent}")
        print(f"Prediction: {pred_val}")
        status = "POSITIVE (+)" if np.any(pred_val == 1) else "NEGATIVE (-)"
        print("Diagnosis:", status)
    except Exception as e:
        print(f"Detection failed for scale 1/{scale}: {e}")

# 2. Configure Model and Run Prediction
# We use the fine-tuned Chagas detection model weights
combined_ckpt_path = './FINETUNED_WEIGHTS/checkpoint_linear_eval_combined_data_20260415-200225.pth'

if not os.path.exists(combined_ckpt_path):
    # Try an alternative path if the one above doesn't exist
    print("Checkpoint not found")
    exit()

print(f"Using checkpoint: {combined_ckpt_path}")

try:
    # Run the detection function
    result = detect_disease(
        ecg_input=ecg_input,
        combined_ckpt_path=combined_ckpt_path, 
        num_classes=1,  # 1 for binary classification (Chagas Positive/Negative)
        device='cuda' if torch.cuda.is_available() else 'cpu',
        threshold=0.5   # Probability threshold for positive prediction
    )
    
    # 3. Display Inference Results
    print("\n" + "="*45)
    print("      CHAGAS DISEASE DETECTION RESULTS")
    print("="*45)
    
    # Extract probability and prediction from the result dictionary
    # The detect_disease function returns values in numpy arrays for batch processing
    prob_val = result['probability']
    pred_val = result['prediction']
    
    status = "POSITIVE (+)" if pred_val == 1 else "NEGATIVE (-)"
    
    print(f"Diagnosis Status:    {prob_val}")
    print(f"Confidence Level:    {pred_val}")
    print(f"Diagnosis Status:    {prob_val.flatten()*100}")
    print(f"Confidence Level:    {pred_val.flatten()*100}")
    print("-" * 45)
    

    # Diagnosis Status:    [[0.04599723]]
    # Confidence Level:    [[0]]
    # Diagnosis Status:    [0.04599723]
    # Confidence Level:    [0]


    if pred_val == 1:
        print("ALERT: The analysis indicates a high likelihood")
        print("       of Chagas disease presence in this ECG.")
    else:
        print("INFO: The analysis did not detect significant")
        print("      indicators of Chagas disease.")
        
    print("="*45 + "\n")
    
except Exception as e:
    print(f"\nExecution failed: {e}")
    import traceback
    traceback.print_exc()
    