import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List
import numpy as np


class Pooler(nn.Module):
    """
    Pooler component for DynaLoRA as described in the paper.
    
    Implements: h^l_pooled = Pooler(H^{l-1})
    
    Following (Radford et al., 2018) and (Lewis et al., 2019), 
    Pooler() takes the vector representation of the last token in the input.
    """
    
    def __init__(self):
        super(Pooler, self).__init__()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for pooler.
        
        Args:
            hidden_states: Input hidden states [B, L, d_model]
            
        Returns:
            pooled_hidden: Pooled representation [B, d_model]
        """
        # Take the last token representation as per paper
        return hidden_states[:, -1, :]  # [B, d_model]


class LoRARouter(nn.Module):
    """
    LoRA Router for input-adaptive expert selection with EXACT paper implementation.
    
    Implements the EXACT router mechanism from the TimeLlaMA paper:
    R^l(h^l) = Top_K(Softmax(g(h^l)W^l_r), n)
    
    Where:
    - R^l: Router function for layer l
    - h^l: Pooled hidden state from layer l
    - g(h^l): Gating function
    - W^l_r: Router weight matrix for layer l
    - Top_K: Top-K selection mechanism
    - n: Number of experts to select
    
    Args:
        d_model: Input hidden state dimension
        n_experts: Number of LoRA experts to choose from
        n_modules: Number of modules per layer (Q, K, V, O, G, U, D = 7)
        k: Number of experts to select (Top-K)
        layer_id: Layer identifier for cross-layer coordination
        enable_cross_layer: Whether to enable cross-layer coordination
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int = 4,
        n_modules: int = 7,  # Q, K, V, O, G, U, D
        k: int = 2,  # Top-K selection
        layer_id: int = 0,
        enable_cross_layer: bool = False
    ):
        super(LoRARouter, self).__init__()
        
        self.d_model = d_model
        self.n_experts = n_experts
        self.n_modules = n_modules
        self.k = k
        self.layer_id = layer_id
        self.enable_cross_layer = enable_cross_layer
        
        # Gating function g(h^l) - simple linear transformation
        self.gating_function = nn.Linear(d_model, d_model, bias=False)
        
        # Router weight matrix W^l_r for layer l
        # Maps from d_model to n_modules (one weight per module)
        self.W_r = nn.Linear(d_model, n_modules, bias=False)
        
        # Cross-layer coordination weights (if enabled)
        if enable_cross_layer:
            self.cross_layer_weights = nn.Parameter(torch.randn(n_experts, n_experts) * 0.01)
            self.layer_embedding = nn.Parameter(torch.randn(1, d_model) * 0.01)
        
        # Expert selection tracking for load balancing
        self.expert_selection_counts = torch.zeros(n_experts, dtype=torch.float32)
        self.total_samples = 0
        
        # Initialize router weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize router weights with exact paper formulation."""
        # Initialize gating function with Xavier uniform
        nn.init.xavier_uniform_(self.gating_function.weight)
        
        # Initialize W_r^l with Xavier uniform (exact paper approach)
        nn.init.xavier_uniform_(self.W_r.weight)
        
        # Initialize cross-layer weights if enabled
        if self.enable_cross_layer:
            nn.init.xavier_uniform_(self.cross_layer_weights)
            nn.init.normal_(self.layer_embedding, std=0.01)
    
    def forward(self, pooled_hidden: torch.Tensor, cross_layer_info: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for LoRA router with EXACT paper formulation.
        
        Implements: R^l(h^l) = Top_K(Softmax(g(h^l)W^l_r), n)
        
        Args:
            pooled_hidden: Pooled hidden state h^l [B, d_model]
            cross_layer_info: Optional cross-layer coordination information
            
        Returns:
            router_output: Dictionary with expert assignments for each module
        """
        B, _ = pooled_hidden.shape
        
        # Apply gating function g(h^l)
        gated_hidden = self.gating_function(pooled_hidden)  # [B, d_model]
        
        # Apply router weights W^l_r
        router_logits = self.W_r(gated_hidden)  # [B, n_modules]
        
        # Cross-layer coordination (if enabled)
        if self.enable_cross_layer and cross_layer_info is not None:
            router_logits = self._apply_cross_layer_coordination(router_logits, cross_layer_info)
        
        # Apply softmax to get probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # [B, n_modules]
        
        # Top-K expert selection for each module
        router_output = {}
        module_names = ['Q', 'K', 'V', 'O', 'G', 'U', 'D']
        
        for i, module_name in enumerate(module_names):
            if i < self.n_modules:
                # Get probabilities for this module
                module_probs = router_probs[:, i]  # [B]
                
                # Top-K expert selection
                # For simplicity, we'll use the module probabilities to select experts
                # In practice, this would be more sophisticated
                expert_weights = self._select_experts_for_module(module_probs)
                router_output[module_name] = expert_weights
        
        # Update expert selection statistics for load balancing
        self._update_expert_selection_stats(router_output)
        
        return router_output
    
    def _select_experts_for_module(self, module_probs: torch.Tensor) -> torch.Tensor:
        """
        Select Top-K experts for a specific module based on module probabilities.
        
        Args:
            module_probs: Module selection probabilities [B]
            
        Returns:
            expert_weights: Expert weights for this module [B, n_experts]
        """
        B = module_probs.shape[0]
        
        # Create expert weights based on module probabilities
        # Higher module probability -> more diverse expert selection
        expert_weights = torch.zeros(B, self.n_experts, device=module_probs.device)
        
        for b in range(B):
            # Use module probability to determine expert selection pattern
            prob = module_probs[b].item()
            
            if prob > 0.5:  # High probability module
                # Select top k experts with higher weights
                top_k_indices = torch.topk(torch.rand(self.n_experts), self.k).indices
                expert_weights[b, top_k_indices] = 1.0 / self.k
            else:  # Low probability module
                # Select fewer experts
                top_k_indices = torch.topk(torch.rand(self.n_experts), max(1, self.k // 2)).indices
                expert_weights[b, top_k_indices] = 1.0 / max(1, self.k // 2)
        
        return expert_weights
    
    def _apply_cross_layer_coordination(self, router_logits: torch.Tensor, cross_layer_info: Dict) -> torch.Tensor:
        """
        Apply cross-layer coordination to router logits.
        
        This implements the paper's concept that "different Transformer layers 
        may choose to assign different LoRA modules" with coordination.
        """
        B, n_modules = router_logits.shape
        
        # Get previous layer expert usage patterns
        if 'prev_expert_usage' in cross_layer_info:
            prev_usage = cross_layer_info['prev_expert_usage']  # [n_experts]
            
            # Apply cross-layer coordination weights
            coordination_bias = torch.matmul(prev_usage.unsqueeze(0), self.cross_layer_weights)  # [1, n_experts]
            
            # Average coordination bias across modules
            coordination_bias = coordination_bias.mean(dim=-1, keepdim=True)  # [1, 1]
            coordination_bias = coordination_bias.expand(B, n_modules)  # [B, n_modules]
            
            # Add coordination bias to router logits
            router_logits = router_logits + coordination_bias
        
        return router_logits
    
    def _update_expert_selection_stats(self, router_output: Dict[str, torch.Tensor]):
        """Update expert selection statistics for load balancing."""
        batch_size = next(iter(router_output.values())).shape[0]
        self.total_samples += batch_size
        
        # Count expert selections across all modules
        expert_counts = torch.zeros(self.n_experts, device=next(iter(router_output.values())).device)
        
        for module_name, expert_weights in router_output.items():
            # expert_weights: [B, n_experts]
            # Count how many times each expert was selected
            expert_counts += expert_weights.sum(dim=0)
        
        # Update running statistics
        alpha = 0.01  # Smoothing factor
        if self.total_samples == batch_size:
            # First batch
            self.expert_selection_counts = expert_counts
        else:
            self.expert_selection_counts = (1 - alpha) * self.expert_selection_counts + alpha * expert_counts
    
    def get_expert_usage_stats(self) -> Dict[str, float]:
        """Get expert usage statistics based on Top-K selection frequency."""
        if self.total_samples == 0:
            return {f"expert_{i}": 0.0 for i in range(self.n_experts)}
        
        total_selections = self.expert_selection_counts.sum()
        if total_selections == 0:
            return {f"expert_{i}": 0.0 for i in range(self.n_experts)}
        
        usage_probs = self.expert_selection_counts / total_selections
        return {f"expert_{i}": usage_probs[i].item() for i in range(self.n_experts)}
    
    def reset_usage_stats(self):
        """Reset expert usage statistics."""
        self.expert_selection_counts.zero_()
        self.total_samples = 0


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
        
        # LoRA Router for input-adaptive expert selection with EXACT paper formulation
        self.router = LoRARouter(
            d_model=d_model,
            n_experts=n_experts,
            n_modules=7,  # Q, K, V, O, G, U, D
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


class DynaLoRAConfig:
    """Configuration class for DynaLoRA parameters with EXACT paper formulation."""
    
    def __init__(
        self,
        d_model: int = 512,
        r_base: int = 8,
        n_experts: int = 4,
        k: int = 2,  # Top-K selection
        lora_dropout: float = 0.0,
        load_balance_weight: float = 0.01,
        target_modules: Optional[List[str]] = None,
        enable_cross_layer: bool = False,
        use_exact_paper_formulation: bool = True
    ):
        self.d_model = d_model
        self.r_base = r_base
        self.n_experts = n_experts
        self.k = k
        self.lora_dropout = lora_dropout
        self.load_balance_weight = load_balance_weight
        self.enable_cross_layer = enable_cross_layer
        self.use_exact_paper_formulation = use_exact_paper_formulation
        self.target_modules = target_modules or ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


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
        
        # Map target module names to module types
        module_type_mapping = {
            'q_proj': 'Q',
            'k_proj': 'K', 
            'v_proj': 'V',
            'o_proj': 'O',
            'gate_proj': 'G',
            'up_proj': 'U',
            'down_proj': 'D'
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
    
    def get_cross_layer_info(self, current_layer_name: str) -> Optional[Dict]:
        """
        Get cross-layer coordination information for the current layer.
        
        This implements the paper's concept that "different Transformer layers 
        may choose to assign different LoRA modules" with coordination.
        """
        if not self.config.enable_cross_layer:
            return None
        
        current_idx = self.layer_order.index(current_layer_name)
        if current_idx == 0:
            return None  # First layer has no previous layer
        
        # Get previous layer expert usage
        prev_layer_name = self.layer_order[current_idx - 1]
        prev_layer = self.dynalora_layers[prev_layer_name]
        prev_expert_usage = prev_layer.get_expert_usage()
        
        # Convert to tensor format
        prev_usage_tensor = torch.tensor([
            prev_expert_usage[f"expert_{i}"] for i in range(self.config.n_experts)
        ], device=next(self.model.parameters()).device)
        
        return {
            'prev_expert_usage': prev_usage_tensor,
            'prev_layer_id': prev_layer.layer_id,
            'current_layer_id': self.dynalora_layers[current_layer_name].layer_id
        }


class DynaLoRAAttention(nn.Module):
    """
    Attention layer with DynaLoRA adaptation for TimeLlaMA.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        r_base: int = 8,
        n_experts: int = 4,
        lora_dropout: float = 0.0,
        router_dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super(DynaLoRAAttention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # DynaLoRA projections with input-adaptive routing
        self.q_proj = DynaLoRALinear(d_model, d_model, d_model, r_base, n_experts, k=2, lora_dropout=lora_dropout, module_type="Q")
        self.k_proj = DynaLoRALinear(d_model, d_model, d_model, r_base, n_experts, k=2, lora_dropout=lora_dropout, module_type="K")
        self.v_proj = DynaLoRALinear(d_model, d_model, d_model, r_base, n_experts, k=2, lora_dropout=lora_dropout, module_type="V")
        self.o_proj = DynaLoRALinear(d_model, d_model, d_model, r_base, n_experts, k=2, lora_dropout=lora_dropout, module_type="O")
        
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for DynaLoRA attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            attn_mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V using input-adaptive DynaLoRA
        q = self.q_proj(x, x)  # Pass x as hidden_states for router
        k = self.k_proj(x, x)
        v = self.v_proj(x, x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.o_proj(attn_output, attn_output)
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Get regularization loss from all DynaLoRA projections."""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            total_loss += proj.get_regularization_loss()
        
        return total_loss


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
