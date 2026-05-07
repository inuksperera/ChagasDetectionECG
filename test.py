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
record_path = os.path.join(dirname, 'testing data/negative/20000_hr')
# record_path = os.path.join(dirname, 'testing data/positive/4991')
print(os.path.exists(record_path + '.dat'))
ecg_input = np.empty((1, 5000, 12), dtype=np.float32)
ecg_input[0] = np.random.randn(1, 5000, 12).astype(np.float32)

# Preprocessing matching official pipeline logic (transpose, resample, lead selection)
ecg_input = ecg_input.transpose(0, 2, 1)
ecg_input = np.array([resample(ecg_input[0], 2500, axis=1)])
ecg_input = np.concatenate([ecg_input[:, :2], ecg_input[:, 6:]], axis=1)

print("SHAPE AFTER PROCESSING: " + str(ecg_input.shape))

print("SHAPE AFTER REDUCED LEAD: " + str(ecg_input.shape))



# # apply normalization (Z-score) as expected by the trained model
# if 'negative' in str(record_path):
#     print("Applying normalization")
# ecg_input = normalize_ecg_per_lead(ecg_input)

# 2. Configure Model and Run Prediction
# We use the fine-tuned Chagas detection model weights
combined_ckpt_path = './FINETUNED_WEIGHTS/ejepa.pth'

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
    