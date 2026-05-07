import numpy as np
from detect_disease import detect_disease
import torch
import numpy as np
from detect_disease import detect_disease
import torch
import os
import wfdb
from ecg_data import normalize_ecg_per_lead
from scipy.signal import resample
# 1. Load your ECG data (Shape: [Leads, Time] or [Batch, Leads, Time])
# Example: 12 leads, 5000 time steps
# my_ecg = np.load('my_ecg_sample.npy') 
dummy_data = np.random.randn(1, 8, 2500).astype(np.float32)
dirname = os.path.dirname(__file__)
# record_path = os.path.join(dirname, 'testing data/negative/20000_hr')
record_path = os.path.join(dirname, 'testing data/positive/3629')
print(os.path.exists(record_path + '.dat'))
data = wfdb.rdsamp(record_path)

signals, fields = wfdb.rdsamp(record_path) # (5000, 12)
dummy_data = np.expand_dims(signals, axis=0).astype(np.float32) # (1, 5000, 12)
dummy_data = dummy_data.transpose(0, 2, 1)

# Select only required leads (only uses 8 leads out of 12)
dummy_data = np.concatenate((dummy_data[:, :2, :], dummy_data[:, 6:, :]), axis=1)
dummy_data = resample(dummy_data, 2500, axis=2)
dummy_data = normalize_ecg_per_lead(dummy_data)

# # 2. Run detection
# result = detect_disease(
#     ecg_input=dummy_data,
#     # encoder_ckpt_path='./weights/multiblock_epoch100.pth',
#     # head_ckpt_path='./output/linear_eval/checkpoint.pth', 
#     combined_ckpt_path='/content/drive/MyDrive/ChagasDetectionECG/downstream_tasks/output/linear_eval/checkpoint_linear_eval_final.pth', 
#     num_classes=5, # 1 for binary classification
#     device='cuda'  # or 'cpu'
# )
# print(f"Prediction: {result['prediction']}")
# print(f"Probability: {result['probability']}")

"""
def detect_disease(ecg_input, encoder_ckpt_path=None, head_ckpt_path=None, combined_ckpt_path=None, use_mlp=False, num_classes=1, device='cuda', threshold=0.5):
"""
try:
    # result = detect_disease(
    #     dummy_data, 
    #     encoder_ckpt_path=args.encoder_ckpt,
    #     head_ckpt_path=args.head_ckpt, 
    #     combined_ckpt_path=args.combined_ckpt,
    #     use_mlp=args.use_mlp,
    #     num_classes=args.num_classes,
    #     device=args.device,
    #     threshold=args.threshold
    # )
    result = detect_disease(
    ecg_input=dummy_data,
    # encoder_ckpt_path='./weights/multiblock_epoch100.pth',
    # head_ckpt_path='./output/linear_eval/checkpoint.pth', 
    # combined_ckpt_path='./downstream_tasks/output/linear_eval/checkpoint_linear_eval_final.pth',
    combined_ckpt_path='./FINETUNED_WEIGHTS/ejepa.pth', 
    num_classes=1, # 1 for binary classification
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
    print("\n" + "="*30)
    print("INFERENCE RESULTS")
    print("="*30)
    
    if 'predicted_disease' in result:
            print(f"Predicted Disease: {result['predicted_disease']}")
            print(f"Top 3 Predictions: {result['top_diseases']}")
            print(f"Top 3 Probabilities: {result['top_probs']}")
    else:
            print(f"Prediction indices: {result['prediction']}")
            print(f"Probabilities (Batch x Classes):\n{result['probability']}")
            
    print("="*30 + "\n")
    
except Exception as e:
    print(f"\nExecution failed: {e}")
    import traceback
    traceback.print_exc()