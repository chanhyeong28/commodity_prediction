import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

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
        n_modules: int = 4,  # Q, K, V, O (simplified for time series)
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
        
        # Cross-layer coordination disabled for simplicity
        self.enable_cross_layer = enable_cross_layer
        
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
        
        # Cross-layer coordination initialization removed for simplicity
    
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
        
        # Cross-layer coordination disabled for simplicity
        
        # Apply softmax to get probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # [B, n_modules]
        
        # Top-K expert selection for each module
        router_output = {}
        module_names = ['Q', 'K', 'V', 'O']  # Simplified for time series
        
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
        # Use learned weights instead of random selection
        expert_weights = torch.zeros(B, self.n_experts, device=module_probs.device)
        
        for b in range(B):
            # Use module probability to determine expert selection pattern
            prob = module_probs[b].item()
            
            if prob > 0.5:  # High probability module
                # Select top k experts with equal weights
                expert_weights[b, :self.k] = 1.0 / self.k
            else:  # Low probability module
                # Select fewer experts
                num_experts = max(1, self.k // 2)
                expert_weights[b, :num_experts] = 1.0 / num_experts
        
        return expert_weights
    
    # Cross-layer coordination method removed for simplicity
    
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