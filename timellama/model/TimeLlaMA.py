import torch
import torch.nn as nn
from typing import Optional, List, Union, Dict
from math import sqrt

from timellama.layers import TSEmb, TSEmbConv, TSEmbHybrid, PromptAlignment
from timellama.layers.StandardNorm import Normalize
from timellama.layers.DynaLoRA import DynaLoRAConfig, DynaLoRAWrapper, create_dynalora_model


class FlattenHead(nn.Module):
    """
    Flatten head for output projection, similar to Time-LLM.
    Handles multi-variate time series output projection.
    """
    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ReprogrammingLayer(nn.Module):
    """
    Reprogramming layer for aligning time series embeddings with LLM vocabulary space.
    Adapted from Time-LLM's ReprogrammingLayer for Time-LlaMA's channel-as-token approach.
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_keys: Optional[int] = None, 
        d_llm: Optional[int] = None, 
        attention_dropout: float = 0.1
    ):
        super(ReprogrammingLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_llm = d_llm or d_model
        
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        for proj in [self.query_projection, self.key_projection, self.value_projection, self.out_projection]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self, 
        target_embedding: torch.Tensor, 
        source_embedding: torch.Tensor, 
        value_embedding: torch.Tensor
    ) -> torch.Tensor:
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def reprogramming(
        self, 
        target_embedding: torch.Tensor, 
        source_embedding: torch.Tensor, 
        value_embedding: torch.Tensor
    ) -> torch.Tensor:
        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding


class TimeLlaMA(nn.Module):
    """
    Time-LlaMA model for time series forecasting using LLM backbones.
    
    Key components:
    - Channel-as-token embedding (TSEmb variants)
    - Prompt alignment via cross-attention (PromptAlignment)
    - Reprogramming layer for vocabulary alignment
    - LLM backbone (frozen)
    - Output projection head
    
    This implementation follows Time-LLM's architecture patterns while
    implementing Time-LlaMA's channel-as-token approach.
    """

    def __init__(
        self,
        llm_model,                 # transformers model (e.g., LlamaModel)
        tokenizer,                 # corresponding tokenizer
        d_model: int,
        lookback: int,
        pred_len: int,
        num_channels: Optional[int] = None,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        embedding_type: str = 'linear',  # 'linear', 'conv', 'hybrid'
        use_positional_emb: bool = True,
        use_channel_emb: bool = True,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        use_reprogramming: bool = True,
        description: str = "Time series forecasting task",
        freeze_llm: bool = True,
        # DynaLoRA parameters
        use_dynalora: bool = False,
        dynalora_r_base: int = 8,
        dynalora_n_experts: int = 4,
        dynalora_dropout: float = 0.0,
        dynalora_router_dropout: float = 0.1,
        dynalora_target_modules: Optional[List[str]] = None,
        # Ablation variant parameters
        ablation_variant: str = 'full',
        use_vanilla_lora: bool = False,
        use_adalora: bool = False,
        use_moelora: bool = False
    ):
        super().__init__()
        self.llm = llm_model
        self.tokenizer = tokenizer
        self.input_embeddings = llm_model.get_input_embeddings()
        self.d_model = d_model
        self.lookback = lookback
        self.pred_len = pred_len
        self.num_channels = num_channels
        self.d_ff = d_ff or (4 * d_model)
        self.description = description
        
        # DynaLoRA configuration
        self.use_dynalora = use_dynalora
        self.dynalora_wrapper = None
        
        # Ablation variant configuration
        self.ablation_variant = ablation_variant
        self.use_vanilla_lora = use_vanilla_lora
        self.use_adalora = use_adalora
        self.use_moelora = use_moelora
        
        # Freeze LLM parameters
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        # Initialize embedding based on type
        if embedding_type == 'linear':
            self.ts_emb = TSEmb(
                lookback=lookback, 
                d_model=d_model,
                num_channels=num_channels,
                use_positional_emb=use_positional_emb,
                use_channel_emb=use_channel_emb,
                dropout=dropout
            )
        elif embedding_type == 'conv':
            self.ts_emb = TSEmbConv(
                lookback=lookback, 
                d_model=d_model,
                dropout=dropout
            )
        elif embedding_type == 'hybrid':
            self.ts_emb = TSEmbHybrid(
                lookback=lookback, 
                d_model=d_model,
                num_channels=num_channels,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")
        
        # Prompt alignment
        self.align = PromptAlignment(d_model=d_model, num_heads=num_heads)
        
        # Reprogramming layer for vocabulary alignment
        if use_reprogramming:
            self.reprogramming_layer = ReprogrammingLayer(
                d_model=d_model, 
                n_heads=num_heads, 
                d_ff=self.d_ff, 
                d_llm=self.input_embeddings.embedding_dim,
                attention_dropout=dropout
            )
        else:
            self.reprogramming_layer = None
        
        # Output projection
        if num_channels is not None:
            self.output_projection = FlattenHead(
                n_vars=num_channels, 
                nf=self.d_ff, 
                target_window=pred_len,
                head_dropout=head_dropout
            )
        else:
            self.output_head = nn.Linear(d_model, pred_len)
        
        # Normalization layers
        if num_channels is not None:
            self.normalize_layers = Normalize(num_channels, affine=False)
        else:
            self.normalize_layers = None
        
        # Apply DynaLoRA if enabled
        if self.use_dynalora:
            self._apply_dynalora(
                r_base=dynalora_r_base,
                n_experts=dynalora_n_experts,
                lora_dropout=dynalora_dropout,
                router_dropout=dynalora_router_dropout,
                target_modules=dynalora_target_modules
            )

    def _apply_dynalora(
        self,
        r_base: int = 8,
        n_experts: int = 4,
        lora_dropout: float = 0.0,
        router_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
    ):
        """Apply DynaLoRA to the model."""
        if target_modules is None:
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        
        # Create DynaLoRA configuration
        config = DynaLoRAConfig(
            d_model=self.d_model,
            r_base=r_base,
            n_experts=n_experts,
            lora_dropout=lora_dropout,
            router_dropout=router_dropout,
            target_modules=target_modules,
            temperature=1.0,
            learnable_temperature=True
        )
        
        # Apply DynaLoRA to the model
        self.dynalora_wrapper = DynaLoRAWrapper(self, config)
        self.dynalora_wrapper.apply_dynalora(target_modules)
        
        # Freeze base layers and unfreeze LoRA parameters
        self.dynalora_wrapper.freeze_base_layers()
        self.dynalora_wrapper.unfreeze_lora_parameters()
    
    def get_dynalora_regularization_loss(self) -> torch.Tensor:
        """Get DynaLoRA regularization loss."""
        if self.dynalora_wrapper is not None:
            return self.dynalora_wrapper.get_regularization_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def get_expert_usage_stats(self) -> Dict[str, Dict[str, float]]:
        """Get expert usage statistics from all DynaLoRA layers."""
        if not hasattr(self, 'dynalora_wrapper') or self.dynalora_wrapper is None:
            return {}
        
        stats = {}
        for name, layer in self.dynalora_wrapper.dynalora_layers.items():
            if hasattr(layer, 'get_expert_usage'):
                stats[name] = layer.get_expert_usage()
        return stats
    
    def reset_expert_usage_stats(self):
        """Reset expert usage statistics for all DynaLoRA layers."""
        if hasattr(self, 'dynalora_wrapper') and self.dynalora_wrapper is not None:
            for layer in self.dynalora_wrapper.dynalora_layers.values():
                if hasattr(layer, 'reset_usage_stats'):
                    layer.reset_usage_stats()
    
    def get_expert_usage(self) -> dict:
        """Get expert usage statistics for all DynaLoRA layers."""
        if self.dynalora_wrapper is not None:
            return self.dynalora_wrapper.get_expert_usage()
        return {}

    def calculate_lags(self, x_enc: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """
        Calculate top-k autocorrelation lags for prompt generation.
        Adapted from Time-LLM's lag calculation.
        """
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, top_k, dim=-1)
        return lags

    def generate_prompt(
        self, 
        x_enc: torch.Tensor, 
        pred_len: int, 
        seq_len: int
    ) -> List[str]:
        """
        Generate descriptive prompts for time series forecasting.
        Adapted from Time-LLM's prompt generation.
        """
        B, T, N = x_enc.size()
        x_enc_flat = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        min_values = torch.min(x_enc_flat, dim=1)[0]
        max_values = torch.max(x_enc_flat, dim=1)[0]
        medians = torch.median(x_enc_flat, dim=1).values
        lags = self.calculate_lags(x_enc_flat)
        trends = x_enc_flat.diff(dim=1).sum(dim=1)
        
        prompt = []
        for b in range(x_enc_flat.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(pred_len)} steps given the previous {str(seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)
        
        return prompt

    @torch.no_grad()
    def build_prompt_embeddings(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        """
        Build prompt embeddings from text prompts.
        """
        toks = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )["input_ids"].to(device)
        emb = self.input_embeddings(toks)
        return emb

    def forward(
        self,
        x_enc: torch.Tensor,          # [B, TL, N]
        x_mark_enc: Optional[torch.Tensor] = None,  # Optional time features
        x_dec: Optional[torch.Tensor] = None,       # Optional decoder input
        x_mark_dec: Optional[torch.Tensor] = None,  # Optional decoder time features
        mask: Optional[torch.Tensor] = None,        # Optional attention mask
        use_prompt: bool = True,                    # Whether to use generated prompts
        custom_prompts: Optional[List[str]] = None  # Custom prompts if provided
    ) -> torch.Tensor:
        """
        Forward pass for Time-LlaMA model.
        
        Architecture per paper:
        1. Time Series → Channel-as-Token Embedding → H0
        2. Generate Text Prompts → Prompt Embeddings
        3. Cross-Attention Alignment: H0 + Prompt Embeddings → Aligned H0
        4. Reprogramming Layer (optional): Aligned H0 → LLM-compatible H0
        5. LLM Backbone: ONLY H0 goes through LLM (prompts do NOT go through LLM)
        6. Output Projection: LLM output → Forecast
        
        Key Design Principle: "text prompt is not passed through the Transformer backbone 
        to minimize inference delay" (from paper)
        
        Args:
            x_enc: Input time series [B, TL, N]
            x_mark_enc: Optional encoder time features
            x_dec: Optional decoder input (not used in this implementation)
            x_mark_dec: Optional decoder time features (not used)
            mask: Optional attention mask
            use_prompt: Whether to generate and use prompts
            custom_prompts: Custom prompts if provided
            
        Returns:
            y_hat: Forecasted values [B, N, TP]
        """
        device = x_enc.device
        B, TL, N = x_enc.shape
        
        # Normalize input if normalization is enabled
        if self.normalize_layers is not None:
            x_enc = self.normalize_layers(x_enc, 'norm')
        
        # 1) Channel-as-token embedding
        H0 = self.ts_emb(x_enc)                    # [B, N, d_model]
        
        # 2) Generate and process prompts (for alignment only, NOT for LLM input)
        if use_prompt:
            if custom_prompts is not None:
                prompts = custom_prompts
            else:
                prompts = self.generate_prompt(x_enc, self.pred_len, TL)
            
            prompt_embeddings = self.build_prompt_embeddings(prompts, device)  # [B, P, d_llm]
            
            # 3) Prompt alignment via cross-attention
            # This aligns time series tokens with prompt embeddings but prompts do NOT go through LLM
            H0 = self.align(H0, prompt_embeddings)  # [B, N, d_model]
        
        # 4) Reprogramming layer (if enabled)
        if self.reprogramming_layer is not None:
            # Get word embeddings for vocabulary alignment
            word_embeddings = self.input_embeddings.weight  # [vocab_size, d_llm]
            H0 = self.reprogramming_layer(H0, word_embeddings, word_embeddings)
        
        # 5) LLM encoding
        # Per paper: "text prompt is not passed through the Transformer backbone to minimize inference delay"
        # Only aligned time series tokens go through the LLM backbone
        llm_out = self.llm(inputs_embeds=H0).last_hidden_state  # [B, N, d_llm]
        
        # 6) Output projection
        if hasattr(self, 'output_projection'):
            # Multi-variate case with FlattenHead
            # Reshape for FlattenHead: [B, N, d_llm] -> [B, N, 1, d_llm]
            llm_out_reshaped = llm_out.unsqueeze(2)
            y_hat = self.output_projection(llm_out_reshaped)  # [B, N, TP]
        else:
            # Single-variate case with simple linear head
            y_hat = self.output_head(llm_out)  # [B, N, TP]
        
        # Denormalize output if normalization was applied
        if self.normalize_layers is not None:
            y_hat = self.normalize_layers(y_hat, 'denorm')
        
        return y_hat

    def forecast(
        self, 
        x_enc: torch.Tensor, 
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forecast method for compatibility with Time-LLM interface.
        """
        return self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)


