import torch
import torch.nn as nn
import numpy as np
import os
import sys

# # Ensure we can import from the current directory
# current_dir = os.getcwd()
# sys.path.append(current_dir)

from models import load_encoder
from linear_probe_utils import LinearClassifier

class MLPHead(nn.Module):
    """Simple MLP head for non-linear classification."""
    def __init__(self, feature_dim, hidden_dim=256, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def detect_disease(ecg_input, encoder_ckpt_path=None, head_ckpt_path=None, combined_ckpt_path=None, use_mlp=False, num_classes=1, device=None, threshold=0.5): #changed num_classes to 1 from 5
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    """
    Detects if a single ECG input is positive for a disease.
    
    Args:
        ecg_input: np.ndarray [leads, seq_len] or [batch, leads, seq_len]
        encoder_ckpt_path: str (Optional) - Path to PRETRAINED encoder only (required if combined_ckpt_path not used)
        head_ckpt_path: str (Optional) - Path to trained head only
        combined_ckpt_path: str (Optional) - Path to FULL SAVED MODEL from linear_eval/finetuning
        use_mlp: bool
        num_classes: int
        device: str
        threshold: float
    
    Returns:
        dict: {'prediction': int/list, 'probability': float/list}
    """
    

    from linear_probe_utils import FinetuningClassifier, LinearClassifier
    

    init_ckpt_path = encoder_ckpt_path if encoder_ckpt_path else combined_ckpt_path
    if not init_ckpt_path:
        raise ValueError("Must provide either encoder_ckpt_path or combined_ckpt_path to initialize architecture.")

    print(f"Initializing encoder architecture...")

    encoder, embed_dim = load_encoder(ckpt_dir=init_ckpt_path)
    
    ## Build the Wrapper
    # Use FinetuningClassifier as the container (same as linear_eval/finetuning)
    model = FinetuningClassifier(encoder, embed_dim, num_classes, device=device)
    
    # Replace head if using MLP (FinetuningClassifier uses LinearClassifier by default)
    if use_mlp:

         model.fc = MLPHead(embed_dim, num_classes=num_classes)
    
    model.to(device)
    model.eval()

    ## Load Weights
    if combined_ckpt_path:
        print(f"Loading combined model from {combined_ckpt_path}...")
        checkpoint = torch.load(combined_ckpt_path, map_location=device, weights_only=False)
        # 'save_model' saves in 'model' key
        state_dict = checkpoint.get('model', checkpoint) 
        
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Load status: {msg}")
        
    elif head_ckpt_path:
        print(f"Loading separate head from {head_ckpt_path}...")
 
        checkpoint = torch.load(head_ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        
        # Remove 'fc.' prefix if present (from FinetuningClassifier) -> fitting into model.fc
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('fc.'):
                new_state_dict[k[3:]] = v
            elif not k.startswith('encoder.'):
                new_state_dict[k] = v
        
        try:
            model.fc.load_state_dict(new_state_dict)
        except Exception as e:
            print(f"Compact load failed ({e}), trying full strict=False on wrapper...")
            model.load_state_dict(state_dict, strict=False)

    ## Preprocess Input
    if isinstance(ecg_input, np.ndarray):
        ecg_input = torch.from_numpy(ecg_input).float()
    
    if ecg_input.dim() == 2:
        ecg_input = ecg_input.unsqueeze(0)
    
    ecg_input = ecg_input.to(device)


    required_length = 2500
    current_length = ecg_input.shape[-1]
    
    if current_length != required_length:
        print(f"Resampling input from {current_length} to {required_length}...")
        # torch.nn.functional.interpolate expects [batch, channels, length]
        ecg_input = torch.nn.functional.interpolate(ecg_input, size=required_length, mode='linear', align_corners=False)
    
    ## Inference
    with torch.no_grad():
        # FinetuningClassifier forward passes input -> encoder -> head
        logits = model(ecg_input)
        
        # --- MULTICLASS LOGIC (PTB-XL 5 Superclasses) ---
        CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"] 
        # NORM: Normal
        # MI: Myocardial Infarction
        # STTC: ST/T Change
        # CD: Conduction Disturbance
        # HYP: Hypertrophy

        if num_classes == 5:
            probs = torch.softmax(logits, dim=-1) # (batch, 5)
            predictions = torch.argmax(probs, dim=-1) # (batch,)
            
            topk_probs, topk_indices = torch.topk(probs, k=min(3, num_classes), dim=-1)
            
            result = {
                'prediction': predictions.cpu().numpy(), # Index
                'probability': probs.cpu().numpy(),      # All probs
                'predicted_disease': [CLASS_NAMES[idx] for idx in predictions.cpu().numpy()],
                'top_diseases': [[CLASS_NAMES[i] for i in indices] for indices in topk_indices.cpu().numpy()],
                'top_probs': topk_probs.cpu().numpy()
            }
            return result
        elif num_classes == 1:
            probs = torch.sigmoid(logits)
            predictions = (probs > threshold).int()
            return {
                'prediction': predictions.cpu().numpy(),
                'probability': probs.cpu().numpy()
            }
            # return {
            #     'logits': logits.squeeze(),
            #     'probabilities': probabilities.squeeze(),
            #     'predictions': predictions.squeeze(),
            #     'probability_positive': probabilities.squeeze().item() if probabilities.numel() == 1 else probabilities.squeeze(),
            #     'predicted_class': int(predictions.squeeze().item()) if predictions.numel() == 1 else predictions.squeeze()
            # }

        # --- PRESERVED BINARY LOGIC  ---
        # if num_classes == 1:
        #     probs = torch.sigmoid(logits)
        #     predictions = (probs > threshold).int()
        # else:
        #     # Fallback for other num_classes if not 5
        #     probs = torch.softmax(logits, dim=-1)
        #     predictions = torch.argmax(probs, dim=-1)
            
        # Fallback if num_classes != 5 and != 1 (e.g. 71 all classes)
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)

    return {
        'prediction': predictions.cpu().numpy(),
        'probability': probs.cpu().numpy()
    }













    
"""
if __name__ == "__main__":
    import argparse
    import warnings
    
    # Filter warnings for cleaner output
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Detect disease from ECG using pretrained JEPA model.')
    
    # Model arguments
    parser.add_argument('--encoder_ckpt', type=str, default=None, help='Path to pretrained encoder checkpoint.')
    parser.add_argument('--head_ckpt', type=str, default=None, help='Path to trained classifier head checkpoint.')
    parser.add_argument('--combined_ckpt', type=str, default=None, help='Path to combined model checkpoint (finetuning/linear_eval result).')
    
    # Inference config
    parser.add_argument('--input_file', type=str, default=None, help='Path to .npy file containing ECG data (shape: [leads, steps] or [1, leads, steps]).')
    parser.add_argument('--use_mlp', action='store_true', help='Use MLP head instead of Linear.')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of output classes (default: 5 for PTB-XL superclasses).')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for binary classification.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on.')
    
    # Dummy mode
    parser.add_argument('--dummy', action='store_true', help='Run with dummy data for verification.')

    args = parser.parse_args()

    # Validate arguments
    if not args.dummy and not args.input_file:
        print("Error: Must provide --input_file or use --dummy.")
        sys.exit(1)
        
    if not args.dummy and not (args.encoder_ckpt or args.combined_ckpt):
         print("Error: Must provide either --encoder_ckpt or --combined_ckpt.")
         sys.exit(1)

    # Prepare Data
    if args.dummy:
        print("Running in DUMMY mode...")
        ecg_data = np.random.randn(1, 12, 5000).astype(np.float32)
        
        # Fallback for dummy execution if no weights provided
        if not args.encoder_ckpt and not args.combined_ckpt:
            default_ckpt = './weights/multiblock_epoch100.pth'
            if os.path.exists(default_ckpt):
                args.encoder_ckpt = default_ckpt
            else:
                 print(f"Warning: No checkpoint provided and default {default_ckpt} not found.")
                 # We can't proceed without architecture file usually
                 if not os.path.exists("./weights"):
                     print("Cannot run dummy without at least a weight file for architecture init.")
                     sys.exit(0)
    else:
        print(f"Loading input from {args.input_file}...")
        try:
            ecg_data = np.load(args.input_file)
            print(f"Input shape: {ecg_data.shape}")
        except Exception as e:
            print(f"Failed to load input file: {e}")
            sys.exit(1)

    # Run Inference
    try:
        result = detect_disease(
            ecg_data, 
            encoder_ckpt_path=args.encoder_ckpt,
            head_ckpt_path=args.head_ckpt, 
            combined_ckpt_path=args.combined_ckpt,
            use_mlp=args.use_mlp,
            num_classes=args.num_classes,
            device=args.device,
            threshold=args.threshold
        )
        
        print("\n" + "="*30)
        print("INFERENCE RESULTS")
        print("="*30)
        
        if 'predicted_disease' in result:
             print(f"Predicted Disease: {result['predicted_disease']}")
             print(f"Top 3 Predictions: {result['top_diseases']}")
             print(f"Top 3 Probabilities: {result['top_probs']}")
        else:
             print(f"Prediction: {result['prediction']}")
             print(f"Probability: {result['probability']}")
             
        print("="*30 + "\n")
        
    except Exception as e:
        print(f"\nExecution failed: {e}")
        import traceback
        traceback.print_exc()
"""