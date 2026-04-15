import torch
import torch.nn as nn
import torch.nn.functional as F

class MoLJEPA(nn.Module):
    def __init__(self, base_model, num_layers: int = 12, **mol_kwargs):
        super().__init__()
        self.base = base_model          #  existing JEPA encoder
        hidden_size = getattr(base_model, 'hidden_size', getattr(base_model, 'embed_dim', 768))
        self.mol = PMA(num_layers=num_layers, hidden_dim=hidden_size, **mol_kwargs)
        
    def representation(self, x, return_hidden=False):
        outputs = self.base.representation(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]   # skip embedding layer
        
        if return_hidden:
            return hidden_states
        
        fused = self.mol(hidden_states)
        return fused

    def forward(self, x, return_hidden=False):
        return self.representation(x, return_hidden=return_hidden)

class PMA(nn.Module):
    def __init__(self, num_layers: int, hidden_dim: int, gate_hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, num_layers)
        )
        
        self.proj = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, layer_outputs: list[torch.Tensor], gate_input: torch.Tensor = None):
        if gate_input is None:
            gate_input = layer_outputs[0].mean(dim=1)   # global average pool
        
        weights = F.softmax(self.gate(gate_input), dim=-1)   # [B, num_layers]
        
        fused = torch.zeros_like(layer_outputs[0])
        for i, h_l in enumerate(layer_outputs):
            w_i = weights[:, i].unsqueeze(-1).unsqueeze(-1)
            fused = fused + w_i * h_l
        
        return self.proj(fused.mean(dim=1))   # final [B, D]
