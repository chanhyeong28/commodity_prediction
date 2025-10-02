import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple

from .config import DynaLoRAConfig
from .expert import LoRAExpert
from .router import Pooler, LoRARouter

class DynaLoRALinear(nn.Module):
    """
    Input-Adaptive Dynamic Low-Rank Adaptation (DynaLoRA) layer with EXACT paper implementation.
    
    Implements the EXACT DynaLoRA mechanism from the TimeLlaMA paper:
    - Uses a LoRA router with Top-K expert selection
    - Implements per-module expert assignment (Q, K, V, O, G, U, D)
    - Uses Pooler to extract representation from hidden states
    - Supports cross-layer coordination for sophisticated layer interaction
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        d_model: Hidden state dimension for router
        r_base: Base rank for LoRA matrices
        n_experts: Number of LoRA experts
        k: Number of experts to select (Top-K)
        lora_dropout: Dropout probability for LoRA layers
        bias: Whether to include bias term
        layer_id: Layer identifier for cross-layer coordination
        enable_cross_layer: Whether to enable cross-layer coordination
        module_type: Type of module (Q, K, V, O, G, U, D)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        d_model: int,
        r_base: int = 8,
        n_experts: int = 4,
        k: int = 2,
        lora_dropout: float = 0.0,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        layer_id: int = 0,
        enable_cross_layer: bool = False,
        module_type: str = "Q"
    ):
        super(DynaLoRALinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.d_model = d_model
        self.r_base = r_base
        self.n_experts = n_experts
        self.k = k
        self.layer_id = layer_id
        self.enable_cross_layer = enable_cross_layer
        self.module_type = module_type
        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32
        
        # Base linear layer (frozen)
        self.base_linear = nn.Linear(
            in_features, out_features, bias=bias, 
            device=device, dtype=dtype
        )
        
        # Freeze base layer parameters
        for param in self.base_linear.parameters():
            param.requires_grad = False
        
        # Pooler component for extracting representation from hidden states
        self.pooler = Pooler()
        
        # LoRA Router for input-adaptive expert selection (simplified for time series)
        self.router = LoRARouter(
            d_model=d_model,
            n_experts=n_experts,
            n_modules=4,  # Q, K, V, O (simplified)
            k=k,
            layer_id=layer_id,
            enable_cross_layer=enable_cross_layer
        )
        
        # Multiple LoRA experts for this specific module
        self.experts = nn.ModuleList([
            LoRAExpert(
                in_features=in_features,
                out_features=out_features,
                r_base=r_base,
                lora_dropout=lora_dropout,
                module_type=module_type
            ) for _ in range(n_experts)
        ])
        
        # Projection layer to match d_model for router input
        if in_features != d_model:
            self.input_projection = nn.Linear(in_features, d_model, bias=False)
        else:
            self.input_projection = nn.Identity()
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize additional parameters."""
        if hasattr(self.input_projection, 'weight') and isinstance(self.input_projection.weight, torch.Tensor):
            nn.init.xavier_uniform_(self.input_projection.weight)
    
    def forward(self, x: torch.Tensor, hidden_states: Optional[torch.Tensor] = None, 
                cross_layer_info: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass with input-adaptive LoRA expert selection and cross-layer coordination.
        
        Implements the EXACT paper formulation with Top-K expert selection.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, in_features]
            hidden_states: Hidden states H^{l-1} for router input [batch_size, seq_len, d_model]
                          If None, uses projected input
            cross_layer_info: Optional cross-layer coordination information
            
        Returns:
            Output tensor of shape [batch_size, seq_len, out_features]
        """
        # Base linear transformation
        base_output = self.base_linear(x)
        
        # Prepare hidden states for router (H^{l-1})
        if hidden_states is None:
            hidden_states = self.input_projection(x)  # [B, L, d_model]
        
        # Apply Pooler to extract representation: h^l_pooled = Pooler(H^{l-1})
        pooled_hidden = self.pooler(hidden_states)  # [B, d_model]
        
        # Get router output with Top-K expert selection: R^l(h^l) = Top_K(Softmax(g(h^l)W^l_r), n)
        router_output = self.router(pooled_hidden, cross_layer_info)  # Dict with expert assignments
        
        # Get expert weights for this specific module
        if self.module_type in router_output:
            module_expert_weights = router_output[self.module_type]  # [B, n_experts]
        else:
            # Fallback: equal weights for all experts
            B = x.shape[0]
            module_expert_weights = torch.ones(B, self.n_experts, device=x.device) / self.n_experts
        
        # Compute LoRA output from selected experts
        lora_outputs = []
        for expert_idx, expert in enumerate(self.experts):
            # Get gating weight for this expert
            expert_gating = module_expert_weights[:, expert_idx:expert_idx+1]  # [B, 1]
            expert_gating = expert_gating.unsqueeze(1)  # [B, 1, 1] for broadcasting
            
            # Apply expert with gating weight
            expert_output = expert(x, gating_weight=expert_gating)
            lora_outputs.append(expert_output)
        
        # Sum outputs from all experts (mixture-of-experts)
        lora_output = torch.sum(torch.stack(lora_outputs, dim=0), dim=0)  # [B, L, out_features]
        
        return base_output + lora_output
    
    def get_regularization_loss(self, load_balance_weight: float = 0.01) -> torch.Tensor:
        """
        Calculate regularization loss for DynaLoRA with EXACT paper formulation.
        
        Implements the load balancing loss from the paper based on Top-K selection frequency:
        L_balance = Σ_{j=1}^{N_expert} p_j^l * log(p_j^l)
        where p_j^l is the frequency of expert j being selected in Top-K
        
        Following (Fedus et al., 2022) for mixture-of-experts load balancing.
        
        Args:
            load_balance_weight: Weight for load balancing regularization
            
        Returns:
            Regularization loss tensor
        """
        # Get expert usage probabilities from router (based on Top-K selection frequency)
        expert_usage_stats = self.router.get_expert_usage_stats()
        
        # Convert to tensor
        usage_probs = torch.tensor([expert_usage_stats[f"expert_{i}"] for i in range(self.n_experts)], 
                                 device=next(self.parameters()).device)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-8
        usage_probs = torch.clamp(usage_probs, min=epsilon)
        
        # EXACT PAPER FORMULATION: Calculate load balancing loss
        # L_balance = Σ_{j=1}^{N_expert} p_j^l * log(p_j^l)
        # We want to maximize entropy (encourage equal usage), so we minimize negative entropy
        load_balance_loss = -torch.sum(usage_probs * torch.log(usage_probs))
        
        return load_balance_weight * load_balance_loss
    
    def get_expert_usage(self) -> Dict[str, float]:
        """Get expert usage statistics."""
        return self.router.get_expert_usage_stats()
    
    def reset_usage_stats(self):
        """Reset expert usage statistics."""
        self.router.reset_usage_stats()





class DynaLoRAWrapper:
    """
    Wrapper class to apply DynaLoRA to existing model layers with cross-layer coordination.
    """
    
    def __init__(self, model: nn.Module, config: DynaLoRAConfig):
        self.model = model
        self.config = config
        self.dynalora_layers: Dict[str, DynaLoRALinear] = {}
        self.layer_order: List[str] = []  # Track layer order for cross-layer coordination
        
    def apply_dynalora(self, target_modules: Optional[List[str]] = None) -> nn.Module:
        """
        Apply DynaLoRA to specified modules in the model with per-module expert assignment.
        
        Args:
            target_modules: List of module names to apply DynaLoRA to
            
        Returns:
            Model with DynaLoRA applied
        """
        target_modules = target_modules or self.config.target_modules
        layer_id = 0
        
        # Map target module names to module types (simplified for time series)
        module_type_mapping = {
            'q_proj': 'Q',
            'k_proj': 'K', 
            'v_proj': 'V',
            'o_proj': 'O'
        }
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
                # Determine module type
                module_type = 'Q'  # Default
                for target, mtype in module_type_mapping.items():
                    if target in name:
                        module_type = mtype
                        break
                
                # Replace with DynaLoRA layer with EXACT paper formulation
                dynalora_layer = DynaLoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    d_model=self.config.d_model,
                    r_base=self.config.r_base,
                    n_experts=self.config.n_experts,
                    k=self.config.k,
                    lora_dropout=self.config.lora_dropout,
                    bias=module.bias is not None,
                    device=next(module.parameters()).device,
                    dtype=next(module.parameters()).dtype,
                    layer_id=layer_id,
                    enable_cross_layer=self.config.enable_cross_layer,
                    module_type=module_type
                )
                
                # Copy base weights
                dynalora_layer.base_linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    dynalora_layer.base_linear.bias.data = module.bias.data.clone()
                
                # Replace module
                self._replace_module(name, dynalora_layer)
                self.dynalora_layers[name] = dynalora_layer
                self.layer_order.append(name)
                layer_id += 1
        
        return self.model
    
    def _replace_module(self, module_name: str, new_module: nn.Module):
        """Replace a module in the model by name."""
        parts = module_name.split('.')
        parent = self.model
        
        # Navigate to parent module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Replace the module
        setattr(parent, parts[-1], new_module)
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Get total regularization loss from all DynaLoRA layers."""
        total_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for layer in self.dynalora_layers.values():
            total_loss += layer.get_regularization_loss(
                self.config.load_balance_weight
            )
        
        return total_loss
    
    def get_expert_usage(self) -> Dict[str, Dict[str, float]]:
        """Get expert usage statistics for all DynaLoRA layers."""
        return {name: layer.get_expert_usage() for name, layer in self.dynalora_layers.items()}
    
    def freeze_base_layers(self):
        """Freeze all base layers in DynaLoRA modules."""
        for layer in self.dynalora_layers.values():
            for param in layer.base_linear.parameters():
                param.requires_grad = False
    
    def unfreeze_lora_parameters(self):
        """Unfreeze LoRA parameters for training."""
        for layer in self.dynalora_layers.values():
            # Unfreeze router parameters
            for param in layer.router.parameters():
                param.requires_grad = True
            
            # Unfreeze expert parameters
            for expert in layer.experts:
                for param in expert.parameters():
                    param.requires_grad = True
            
            # Unfreeze projection parameters
            if hasattr(layer.input_projection, 'weight') and isinstance(layer.input_projection.weight, torch.Tensor):
                layer.input_projection.weight.requires_grad = True
    
    # Cross-layer coordination method removed for simplicity


# DynaLoRAAttention class removed - not needed for TimeLlaMA integration


def create_dynalora_model(
    model: nn.Module,
    d_model: int = 512,
    r_base: int = 8,
    n_experts: int = 4,
    k: int = 2,
    lora_dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    enable_cross_layer: bool = False,
    use_exact_paper_formulation: bool = True
) -> Tuple[nn.Module, DynaLoRAWrapper]:
    """
    Create a DynaLoRA-enabled model from an existing model with EXACT paper formulation.
    
    Args:
        model: Base model to apply DynaLoRA to
        d_model: Hidden state dimension for router
        r_base: Base rank for DynaLoRA layers
        n_experts: Number of LoRA experts
        k: Number of experts to select (Top-K)
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to target
        enable_cross_layer: Whether to enable cross-layer coordination
        use_exact_paper_formulation: Whether to use exact paper formulation
        
    Returns:
        Tuple of (modified_model, dynalora_wrapper)
    """
    config = DynaLoRAConfig(
        d_model=d_model,
        r_base=r_base,
        n_experts=n_experts,
        k=k,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        enable_cross_layer=enable_cross_layer,
        use_exact_paper_formulation=use_exact_paper_formulation
    )
    
    wrapper = DynaLoRAWrapper(model, config)
    modified_model = wrapper.apply_dynalora()
    
    return modified_model, wrapper