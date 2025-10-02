import torch
import torch.nn as nn
import math

class LoRAExpert(nn.Module):
    """
    Individual LoRA expert module for a specific Transformer module.
    
    Each expert is specialized for one specific module (Q, K, V, O, G, U, or D).
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r_base: int = 8,
        lora_dropout: float = 0.0,
        module_type: str = "Q"
    ):
        super(LoRAExpert, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r_base = r_base
        self.module_type = module_type
        
        # LoRA matrices for this specific module
        self.lora_A = nn.Parameter(torch.empty(r_base, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r_base))
        
        # Dropout
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize LoRA parameters."""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # Initialize B with zeros (standard LoRA practice)
        nn.init.zeros_(self.lora_B)
    
    def forward(
        self, 
        x: torch.Tensor, 
        gating_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for LoRA expert.
        
        Args:
            x: Input tensor [B, L, in_features]
            gating_weight: Gating weight for this expert [B, L, 1]
            
        Returns:
            LoRA output [B, L, out_features]
        """
        # Apply LoRA transformation: x @ A.T @ B.T
        lora_output = self.dropout(x) @ self.lora_A.T  # [B, L, r_base]
        lora_output = lora_output @ self.lora_B.T      # [B, L, out_features]
        
        # Scale by gating weight
        lora_output = lora_output * gating_weight
        
        return lora_output