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
        
        # DynaLoRA configuration
        self.use_dynalora = use_dynalora
        self.dynalora_wrapper = None
        self.enable_cross_layer = dynalora_target_modules is not None and len(dynalora_target_modules) > 1
        
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
        self.align = PromptAlignment(
            d_model=d_model, 
            d_llm=self.input_embeddings.embedding_dim,
            num_heads=num_heads
        )
        
        # Projection layer to map d_model to d_llm for LLM input
        self.llm_projection = nn.Linear(d_model, self.input_embeddings.embedding_dim)
        
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
                nf=self.input_embeddings.embedding_dim,  # Use d_llm instead of d_ff
                target_window=pred_len,
                head_dropout=head_dropout
            )
        else:
            self.output_head = nn.Linear(self.input_embeddings.embedding_dim, pred_len)  # Use d_llm instead of d_model
        
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
        
        # Create DynaLoRA configuration with EXACT paper formulation
        config = DynaLoRAConfig(
            d_model=self.d_model,
            r_base=r_base,
            n_experts=n_experts,
            k=2,  # Top-K selection
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            enable_cross_layer=self.enable_cross_layer,
            use_exact_paper_formulation=True
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



    @torch.no_grad()
    def build_prompt_embeddings(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        """
        Build prompt embeddings from given text prompts using frozen LLM components.
        
        This method implements the text prompt embedding process from the TimeLlaMA paper:
        
        Process Flow:
        1. Given Text Prompts → LLM Tokenizer → Multiple Tokens
        2. Multiple Tokens → Frozen Embedding Layer → Multiple Embeddings (HP,0)
        3. HP,0 used as keys/values in cross-attention with time series tokens (H0) as queries
        
        Key Design Principles:
        - Uses frozen LLM tokenizer (no training required)
        - Uses frozen LLM embedding layer (no training required)  
        - One text prompt produces multiple embedding vectors
        - Embeddings serve as context for cross-attention alignment
        - Prompts do NOT go through the LLM backbone (only used for alignment)
        
        Args:
            prompts: List of given text prompts (one per batch sample)
            device: Device to place embeddings on
            
        Returns:
            prompt_embeddings: [B, P, d_llm] where P = number of tokens per prompt
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
    
    def debug_prompt_tokenization(self, prompts: List[str], device: torch.device) -> Dict:
        """
        Debug method to visualize the prompt tokenization and embedding process.
        
        This method helps understand how given text prompts are processed:
        1. Shows the original text prompts
        2. Shows tokenization results (tokens per prompt)
        3. Shows embedding dimensions
        4. Demonstrates the "One text → Multiple embeddings" principle
        
        Args:
            prompts: List of given text prompts
            device: Device to process on
            
        Returns:
            Dictionary containing debugging information
        """
        # Tokenize prompts
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        
        # Get embeddings
        embeddings = self.input_embeddings(input_ids)
        
        # Decode tokens for visualization
        decoded_tokens = []
        for i in range(len(prompts)):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            # Remove padding tokens
            valid_tokens = [token for j, token in enumerate(tokens) if attention_mask[i][j] == 1]
            decoded_tokens.append(valid_tokens)
        
        debug_info = {
            "original_prompts": prompts,
            "num_prompts": len(prompts),
            "tokens_per_prompt": [len(tokens) for tokens in decoded_tokens],
            "total_tokens": sum(len(tokens) for tokens in decoded_tokens),
            "embedding_shape": embeddings.shape,  # [B, P, d_llm]
            "embedding_dim": embeddings.shape[-1],
            "sample_tokens": decoded_tokens[0][:10] if decoded_tokens else [],  # First 10 tokens of first prompt
            "tokenization_summary": {
                "one_text_becomes_multiple_tokens": True,
                "one_token_becomes_one_embedding": True,
                "result": "One text → Multiple tokens → Multiple embeddings",
                "frozen_components": ["tokenizer", "embedding_layer"],
                "trainable_components": ["cross_attention_weights"]
            }
        }
        
        return debug_info

    def forward(
        self,
        x_enc: torch.Tensor,          # [B, TL, N]
        x_mark_enc: Optional[torch.Tensor] = None,  # Optional time features
        x_dec: Optional[torch.Tensor] = None,       # Optional decoder input
        x_mark_dec: Optional[torch.Tensor] = None,  # Optional decoder time features
        mask: Optional[torch.Tensor] = None,        # Optional attention mask
        prompts: Optional[List[str]] = None         # Text prompts (one per batch sample)
    ) -> torch.Tensor:
        """
        Forward pass for Time-LlaMA model.
        
        Architecture per paper:
        1. Time Series → Channel-as-Token Embedding → H0 [B, N, d_model]
        2. Given Text Prompts → Tokenize → Embed → HP,0 [B, P, d_llm]
        3. Cross-Attention Alignment: H0 (queries) + HP,0 (keys/values) → Aligned H0
        4. Reprogramming Layer (optional): Aligned H0 → LLM-compatible H0
        5. LLM Backbone: ONLY H0 goes through LLM (prompts do NOT go through LLM)
        6. Output Projection: LLM output → Forecast
        
        Text Prompt Embedding Process (Key Innovation):
        - One text prompt per batch sample (not per channel)
        - Given Text → LLM Tokenizer → Multiple tokens (e.g., 50-200 tokens)
        - Multiple tokens → Frozen Embedding Layer → Multiple embeddings HP,0
        - HP,0 serves as keys/values in cross-attention with H0 as queries
        - Result: Rich context alignment without passing prompts through LLM
        
        Key Design Principle: "text prompt is not passed through the Transformer backbone 
        to minimize inference delay" (from paper)
        
        Args:
            x_enc: Input time series [B, TL, N]
            x_mark_enc: Optional encoder time features
            x_dec: Optional decoder input (not used in this implementation)
            x_mark_dec: Optional decoder time features (not used)
            mask: Optional attention mask
            prompts: Text prompts (one per batch sample) - required for modality alignment
            
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
        
        # 2) Process given prompts (for alignment only, NOT for LLM input)
        if prompts is not None:
            # KEY: Text Prompt Embedding Process (per TimeLlaMA paper)
            # Given text prompts → LLM Tokenizer → Multiple tokens → Frozen Embedding Layer → Multiple embeddings
            # Result: HP,0 [B, P, d_llm] where P = number of tokens per prompt
            prompt_embeddings = self.build_prompt_embeddings(prompts, device)  # [B, P, d_llm]
            
            # 3) Prompt alignment via cross-attention
            # H0 (time series tokens) as queries, HP,0 (prompt embeddings) as keys/values
            # This aligns time series tokens with prompt context but prompts do NOT go through LLM
            H0 = self.align(H0, prompt_embeddings)  # [B, N, d_model]
        
        # 4) Reprogramming layer (if enabled)
        if self.reprogramming_layer is not None:
            # Get word embeddings for vocabulary alignment
            word_embeddings = self.input_embeddings.weight  # [vocab_size, d_llm]
            H0 = self.reprogramming_layer(H0, word_embeddings, word_embeddings)
        
        # 5) Project to LLM dimension
        # H0 is [B, N, d_model], need to project to [B, N, d_llm] for LLM input
        H0_llm = self.llm_projection(H0)  # [B, N, d_llm]
        
        # 6) LLM encoding
        # Per paper: "text prompt is not passed through the Transformer backbone to minimize inference delay"
        # Only aligned time series tokens go through the LLM backbone
        llm_out = self.llm(inputs_embeds=H0_llm).last_hidden_state  # [B, N, d_llm]
        
        # 7) Output projection
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
        x_mark_dec: Optional[torch.Tensor] = None,
        prompts: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Forecast method for compatibility with Time-LLM interface.
        """
        return self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec, prompts=prompts)


