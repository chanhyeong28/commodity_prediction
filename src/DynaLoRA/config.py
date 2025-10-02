from typing import Optional, List

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
        self.target_modules = target_modules or ['q_proj', 'k_proj', 'v_proj', 'o_proj']