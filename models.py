import torch
from ecg_jepa import ecg_jepa

def load_encoder(ckpt_dir, leads=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint first to infer architecture parameters
    ckpt = torch.load(ckpt_dir, weights_only=False, map_location=device)
    
    # Try to find the state dict in common keys
    state_dict = ckpt.get('encoder', ckpt.get('model', ckpt))
    
    # Dynamically infer the capacity 'c' (number of leads model was trained for)
    # the number of tokens in pos_embed is c * p, where p=50 is standard
    inferred_c = 12 # Default
    if 'pos_embed' in state_dict:
        pos_tokens = state_dict['pos_embed'].shape[0]
        inferred_c = pos_tokens // 50
        print(f"Detected checkpoint capacity: {inferred_c} leads")
    elif any(k.startswith('encoder.pos_embed') for k in state_dict.keys()):
        # Handle case where keys have 'encoder.' prefix
        key = [k for k in state_dict.keys() if k.endswith('pos_embed')][0]
        pos_tokens = state_dict[key].shape[0]
        inferred_c = pos_tokens // 50
        print(f"Detected checkpoint capacity: {inferred_c} leads")

    if leads is None:
        # Default leads list to match inferred capacity
        leads = list(range(inferred_c))

    params = {
        'encoder_embed_dim': 768,
        'encoder_depth': 12,
        'encoder_num_heads': 16,
        'predictor_embed_dim': 384,
        'predictor_depth': 6,
        'predictor_num_heads': 12,
        'c': inferred_c,
        'pos_type': 'sincos',
        'mask_scale': (0, 0),
        'leads': leads
    }
    
    # Initialize encoder with the capacity that matches the weights
    encoder = ecg_jepa(**params).encoder
    
    # Clean up state dict if it comes from a combined model (strip 'encoder.' prefix)
    new_state_dict = {}
    has_prefix = any(k.startswith('encoder.') for k in state_dict.keys())
    
    for k, v in state_dict.items():
        if has_prefix:
            if k.startswith('encoder.'):
                new_state_dict[k[8:]] = v
        else:
            new_state_dict[k] = v
            
    # Load weights
    msg = encoder.load_state_dict(new_state_dict, strict=False)
    print(f"Encoder load status: {msg}")
    
    embed_dim = 768
    return encoder, embed_dim
