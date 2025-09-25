import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List
import numpy as np


class LoRARouter(nn.Module):
    """
    LoRA Router for input-adaptive expert selection.
    
    Implements the router mechanism described in the TimeLlaMA paper:
    g_m,l = softmax(W_r^l * H^{l-1} / temperature)
    
    Args:
        d_model: Input hidden state dimension
        n_experts: Number of LoRA experts to choose from
        n_modules: Number of modules per layer (Q, K, V, O, G, U, D = 7)
        temperature: Initial temperature for softmax gating (can be learnable)
        dropout: Dropout rate for router network
        learnable_temperature: Whether temperature is learnable
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int = 4,
        n_modules: int = 7,  # Q, K, V, O, G, U, D
        temperature: float = 1.0,
        dropout: float = 0.1,
        learnable_temperature: bool = True
    ):
        super(LoRARouter, self).__init__()
        
        self.d_model = d_model
        self.n_experts = n_experts
        self.n_modules = n_modules
        self.learnable_temperature = learnable_temperature
        
        # Router network: maps hidden states to expert selection weights
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_experts * n_modules)
        )
        
        # Learnable temperature parameter
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        else:
            self.register_buffer('temperature', torch.tensor(temperature, dtype=torch.float32))
        
        # Expert usage tracking
        self.expert_usage_count = torch.zeros(n_experts, dtype=torch.float32)
        self.total_samples = 0
        
        # Initialize router weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize router weights."""
        for module in self.router:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LoRA router.
        
        Args:
            hidden_states: Input hidden states [B, L, d_model]
            
        Returns:
            gating_weights: Expert selection weights [B, L, n_experts, n_modules]
        """
        B, L, _ = hidden_states.shape
        
        # Compute router logits
        router_logits = self.router(hidden_states)  # [B, L, n_experts * n_modules]
        
        # Reshape to separate experts and modules
        router_logits = router_logits.view(B, L, self.n_experts, self.n_modules)
        
        # Apply temperature scaling
        router_logits = router_logits / self.temperature
        
        # Compute gating weights using softmax
        gating_weights = F.softmax(router_logits, dim=2)  # [B, L, n_experts, n_modules]
        
        # Update expert usage statistics
        self._update_expert_usage(gating_weights)
        
        return gating_weights
    
    def _update_expert_usage(self, gating_weights: torch.Tensor):
        """Update expert usage statistics."""
        # gating_weights: [B, L, n_experts, n_modules]
        # Average over batch, sequence, and modules to get per-expert usage
        expert_usage = gating_weights.mean(dim=(0, 1, 3))  # [n_experts]
        
        # Update running statistics
        batch_size = gating_weights.shape[0]
        self.total_samples += batch_size
        
        # Exponential moving average for expert usage
        alpha = 0.01  # Smoothing factor
        if self.total_samples == batch_size:
            # First batch
            self.expert_usage_count = expert_usage.sum(dim=0)
        else:
            self.expert_usage_count = (1 - alpha) * self.expert_usage_count + alpha * expert_usage.sum(dim=0)
    
    def get_expert_usage_stats(self) -> Dict[str, float]:
        """Get expert usage statistics."""
        if self.total_samples == 0:
            return {f"expert_{i}": 0.0 for i in range(self.n_experts)}
        
        total_usage = self.expert_usage_count.sum()
        if total_usage == 0:
            return {f"expert_{i}": 0.0 for i in range(self.n_experts)}
        
        usage_probs = self.expert_usage_count / total_usage
        return {f"expert_{i}": usage_probs[i].item() for i in range(self.n_experts)}
    
    def reset_usage_stats(self):
        """Reset expert usage statistics."""
        self.expert_usage_count.zero_()
        self.total_samples = 0


class LoRAExpert(nn.Module):
    """
    Individual LoRA expert module.
    
    Each expert contains LoRA matrices for all modules (Q, K, V, O, G, U, D).
    """
    
    def __init__(
        self,
        d_model: int,
        r_base: int = 8,
        n_modules: int = 7,
        lora_dropout: float = 0.0
    ):
        super(LoRAExpert, self).__init__()
        
        self.d_model = d_model
        self.r_base = r_base
        self.n_modules = n_modules
        
        # Create LoRA matrices for each module
        self.lora_A = nn.Parameter(torch.empty(n_modules, r_base, d_model))
        self.lora_B = nn.Parameter(torch.empty(n_modules, d_model, r_base))
        
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
        module_idx: int,
        gating_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for LoRA expert.
        
        Args:
            x: Input tensor [B, L, d_model]
            module_idx: Index of the module (0-6 for Q, K, V, O, G, U, D)
            gating_weight: Gating weight for this expert [B, L, 1]
            
        Returns:
            LoRA output [B, L, d_model]
        """
        # Get LoRA matrices for the specific module
        A = self.lora_A[module_idx]  # [r_base, d_model]
        B = self.lora_B[module_idx]  # [d_model, r_base]
        
        # Apply LoRA transformation
        lora_output = self.dropout(x) @ A.T  # [B, L, r_base]
        lora_output = lora_output @ B.T      # [B, L, d_model]
        
        # Scale by gating weight
        lora_output = lora_output * gating_weight
        
        return lora_output


class DynaLoRALinear(nn.Module):
    """
    Input-Adaptive Dynamic Low-Rank Adaptation (DynaLoRA) layer.
    
    Implements the true DynaLoRA mechanism from the TimeLlaMA paper:
    - Uses a LoRA router to condition expert selection on input
    - Implements mixture-of-experts architecture
    - Dynamically assigns different LoRA modules based on input characteristics
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        d_model: Hidden state dimension for router
        r_base: Base rank for LoRA matrices
        n_experts: Number of LoRA experts
        lora_dropout: Dropout probability for LoRA layers
        router_dropout: Dropout probability for router
        bias: Whether to include bias term
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        d_model: int,
        r_base: int = 8,
        n_experts: int = 4,
        lora_dropout: float = 0.0,
        router_dropout: float = 0.1,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        temperature: float = 1.0,
        learnable_temperature: bool = True
    ):
        super(DynaLoRALinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.d_model = d_model
        self.r_base = r_base
        self.n_experts = n_experts
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
        
        # LoRA Router for input-adaptive expert selection
        self.router = LoRARouter(
            d_model=d_model,
            n_experts=n_experts,
            n_modules=1,  # Single module per layer
            dropout=router_dropout,
            temperature=temperature,
            learnable_temperature=learnable_temperature
        )
        
        # Multiple LoRA experts
        self.experts = nn.ModuleList([
            LoRAExpert(
                d_model=in_features,  # Use in_features for expert dimension
                r_base=r_base,
                n_modules=1,
                lora_dropout=lora_dropout
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
        if hasattr(self.input_projection, 'weight'):
            nn.init.xavier_uniform_(self.input_projection.weight)
    
    def forward(self, x: torch.Tensor, hidden_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with input-adaptive LoRA expert selection.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, in_features]
            hidden_states: Hidden states for router input [batch_size, seq_len, d_model]
                          If None, uses projected input
            
        Returns:
            Output tensor of shape [batch_size, seq_len, out_features]
        """
        # Base linear transformation
        base_output = self.base_linear(x)
        
        # Prepare hidden states for router
        if hidden_states is None:
            hidden_states = self.input_projection(x)  # [B, L, d_model]
        
        # Get gating weights from router
        gating_weights = self.router(hidden_states)  # [B, L, n_experts, 1]
        
        # Compute LoRA output from all experts
        lora_outputs = []
        for expert_idx, expert in enumerate(self.experts):
            # Get gating weight for this expert
            expert_gating = gating_weights[:, :, expert_idx:expert_idx+1, 0:1]  # [B, L, 1, 1]
            expert_gating = expert_gating.squeeze(-1)  # [B, L, 1]
            
            # Apply expert
            expert_output = expert(x, module_idx=0, gating_weight=expert_gating)
            lora_outputs.append(expert_output)
        
        # Sum outputs from all experts
        lora_output = torch.sum(torch.stack(lora_outputs, dim=0), dim=0)  # [B, L, in_features]
        
        # Project to output dimension if needed
        if self.in_features != self.out_features:
            # Create a simple projection layer for this
            if not hasattr(self, 'output_projection'):
                self.output_projection = nn.Linear(
                    self.in_features, self.out_features, bias=False
                ).to(device=x.device, dtype=x.dtype)
                nn.init.xavier_uniform_(self.output_projection.weight)
            lora_output = self.output_projection(lora_output)
        
        return base_output + lora_output
    
    def get_regularization_loss(self, load_balance_weight: float = 0.01) -> torch.Tensor:
        """
        Calculate regularization loss for DynaLoRA.
        
        Implements the load balancing loss from the paper:
        L_balance = Σ_{l=1}^L Σ_{j=1}^{N_expert} p_j^l * log(p_j^l)
        where p_j^l = (1/N_B) * Σ_{i=1}^{N_B} g_{j,l}^i
        
        Args:
            load_balance_weight: Weight for load balancing regularization
            
        Returns:
            Regularization loss tensor
        """
        # Get expert usage probabilities from router
        expert_usage_stats = self.router.get_expert_usage_stats()
        
        # Convert to tensor
        usage_probs = torch.tensor([expert_usage_stats[f"expert_{i}"] for i in range(self.n_experts)], 
                                 device=next(self.parameters()).device)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-8
        usage_probs = torch.clamp(usage_probs, min=epsilon)
        
        # Calculate load balancing loss: -Σ p_j * log(p_j)
        # Negative because we want to maximize entropy (encourage equal usage)
        load_balance_loss = -torch.sum(usage_probs * torch.log(usage_probs))
        
        return load_balance_weight * load_balance_loss
    
    def get_expert_usage(self) -> Dict[str, float]:
        """Get expert usage statistics."""
        return self.router.get_expert_usage_stats()
    
    def reset_usage_stats(self):
        """Reset expert usage statistics."""
        self.router.reset_usage_stats()


class DynaLoRAConfig:
    """Configuration class for DynaLoRA parameters."""
    
    def __init__(
        self,
        d_model: int = 512,
        r_base: int = 8,
        n_experts: int = 4,
        lora_dropout: float = 0.0,
        router_dropout: float = 0.1,
        load_balance_weight: float = 0.01,
        target_modules: Optional[List[str]] = None,
        enable_input_adaptive: bool = True,
        temperature: float = 1.0,
        learnable_temperature: bool = True
    ):
        self.d_model = d_model
        self.r_base = r_base
        self.n_experts = n_experts
        self.lora_dropout = lora_dropout
        self.router_dropout = router_dropout
        self.load_balance_weight = load_balance_weight
        self.temperature = temperature
        self.learnable_temperature = learnable_temperature
        self.target_modules = target_modules or ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        self.enable_input_adaptive = enable_input_adaptive


class DynaLoRAWrapper:
    """
    Wrapper class to apply DynaLoRA to existing model layers.
    """
    
    def __init__(self, model: nn.Module, config: DynaLoRAConfig):
        self.model = model
        self.config = config
        self.dynalora_layers: Dict[str, DynaLoRALinear] = {}
        
    def apply_dynalora(self, target_modules: Optional[List[str]] = None) -> nn.Module:
        """
        Apply DynaLoRA to specified modules in the model.
        
        Args:
            target_modules: List of module names to apply DynaLoRA to
            
        Returns:
            Model with DynaLoRA applied
        """
        target_modules = target_modules or self.config.target_modules
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
                # Replace with DynaLoRA layer
                dynalora_layer = DynaLoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    d_model=self.config.d_model,
                    r_base=self.config.r_base,
                    n_experts=self.config.n_experts,
                    lora_dropout=self.config.lora_dropout,
                    router_dropout=self.config.router_dropout,
                    bias=module.bias is not None,
                    device=next(module.parameters()).device,
                    dtype=next(module.parameters()).dtype,
                    temperature=self.config.temperature,
                    learnable_temperature=self.config.learnable_temperature
                )
                
                # Copy base weights
                dynalora_layer.base_linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    dynalora_layer.base_linear.bias.data = module.bias.data.clone()
                
                # Replace module
                self._replace_module(name, dynalora_layer)
                self.dynalora_layers[name] = dynalora_layer
        
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
            if hasattr(layer.input_projection, 'weight'):
                layer.input_projection.weight.requires_grad = True


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
        self.q_proj = DynaLoRALinear(d_model, d_model, d_model, r_base, n_experts, lora_dropout, router_dropout, temperature=1.0, learnable_temperature=True)
        self.k_proj = DynaLoRALinear(d_model, d_model, d_model, r_base, n_experts, lora_dropout, router_dropout, temperature=1.0, learnable_temperature=True)
        self.v_proj = DynaLoRALinear(d_model, d_model, d_model, r_base, n_experts, lora_dropout, router_dropout, temperature=1.0, learnable_temperature=True)
        self.o_proj = DynaLoRALinear(d_model, d_model, d_model, r_base, n_experts, lora_dropout, router_dropout, temperature=1.0, learnable_temperature=True)
        
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
    lora_dropout: float = 0.0,
    router_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    temperature: float = 1.0,
    learnable_temperature: bool = True
) -> Tuple[nn.Module, DynaLoRAWrapper]:
    """
    Create a DynaLoRA-enabled model from an existing model.
    
    Args:
        model: Base model to apply DynaLoRA to
        d_model: Hidden state dimension for router
        r_base: Base rank for DynaLoRA layers
        n_experts: Number of LoRA experts
        lora_dropout: Dropout probability for LoRA layers
        router_dropout: Dropout probability for router
        target_modules: List of module names to target
        
    Returns:
        Tuple of (modified_model, dynalora_wrapper)
    """
    config = DynaLoRAConfig(
        d_model=d_model,
        r_base=r_base,
        n_experts=n_experts,
        lora_dropout=lora_dropout,
        router_dropout=router_dropout,
        target_modules=target_modules,
        temperature=temperature,
        learnable_temperature=learnable_temperature
    )
    
    wrapper = DynaLoRAWrapper(model, config)
    modified_model = wrapper.apply_dynalora()
    
    return modified_model, wrapper
