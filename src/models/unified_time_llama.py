"""
Unified Time-LlaMA Model

This module consolidates the Time-LlaMA architecture into a single, production-ready
model that combines patch-based embeddings, cross-attention for GraphRAG context,
and PEFT/LoRA for efficient fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import math
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


@dataclass
class TimeLlaMAConfig:
    """Configuration for Time-LlaMA model"""
    # Model architecture
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 256
    
    # Patch embedding
    patch_embed_dim: int = 64
    patch_size: int = 7
    
    # Cross-attention for GraphRAG
    context_dim: int = 64
    context_heads: int = 4
    
    # PEFT/LoRA settings
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Output settings
    num_horizons: int = 4
    output_dim: int = 1


class PatchEmbedding(nn.Module):
    """Patch embedding for time series windows"""
    
    def __init__(self, patch_size: int, d_model: int, patch_embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Simple CNN-based patch embedding
        self.conv = nn.Conv1d(1, patch_embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(patch_embed_dim, d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, patch_size] - time series patches
        Returns:
            [batch_size, seq_len, d_model] - patch embeddings
        """
        batch_size, seq_len, patch_size = x.shape
        
        # Reshape for convolution
        x = x.view(batch_size * seq_len, 1, patch_size)
        
        # Apply convolution and pooling
        x = self.conv(x)  # [batch_size * seq_len, patch_embed_dim, 1]
        x = self.pool(x)  # [batch_size * seq_len, patch_embed_dim, 1]
        x = x.squeeze(-1)  # [batch_size * seq_len, patch_embed_dim]
        
        # Project to model dimension
        x = self.projection(x)  # [batch_size * seq_len, d_model]
        x = self.dropout(x)
        
        # Reshape back
        x = x.view(batch_size, seq_len, self.d_model)
        
        return x


class CrossAttentionAdapter(nn.Module):
    """Cross-attention adapter for GraphRAG context integration"""
    
    def __init__(self, d_model: int, context_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.context_dim = context_dim
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(context_dim, d_model)
        self.v_proj = nn.Linear(context_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, time_tokens: torch.Tensor, context_tokens: torch.Tensor, 
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            time_tokens: [batch_size, seq_len, d_model] - time series tokens
            context_tokens: [batch_size, context_len, context_dim] - GraphRAG context
            context_mask: [batch_size, context_len] - context attention mask
        Returns:
            [batch_size, seq_len, d_model] - enhanced time tokens
        """
        batch_size, seq_len, d_model = time_tokens.shape
        context_len = context_tokens.shape[1]
        
        # Compute queries, keys, values
        q = self.q_proj(time_tokens)  # [batch_size, seq_len, d_model]
        k = self.k_proj(context_tokens)  # [batch_size, context_len, d_model]
        v = self.v_proj(context_tokens)  # [batch_size, context_len, d_model]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, context_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, context_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply context mask if provided
        if context_mask is not None:
            context_mask = context_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, context_len]
            scores = scores.masked_fill(context_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model] - input tokens
            attention_mask: [batch_size, seq_len] - attention mask
        Returns:
            [batch_size, seq_len, d_model] - output tokens
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class UnifiedTimeLlaMA(nn.Module):
    """
    Unified Time-LlaMA model for commodity forecasting with GraphRAG integration.
    
    This model combines:
    1. Patch-based embeddings for time series
    2. Cross-attention for GraphRAG context
    3. Transformer backbone for sequence modeling
    4. PEFT/LoRA for efficient fine-tuning
    """
    
    def __init__(self, config: TimeLlaMAConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_size=config.patch_size,
            d_model=config.d_model,
            patch_embed_dim=config.patch_embed_dim
        )
        
        # Cross-attention adapter for GraphRAG
        self.cross_attention = CrossAttentionAdapter(
            d_model=config.d_model,
            context_dim=config.context_dim,
            n_heads=config.context_heads,
            dropout=config.dropout
        )
        
        # Transformer backbone
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout
            )
            for _ in range(config.n_layers)
        ])
        
        # Output heads
        self.forecast_head = nn.Linear(config.d_model, config.num_horizons * config.output_dim)
        self.embedding_head = nn.Linear(config.d_model, config.patch_embed_dim)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(config.max_seq_len, config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, patch_tokens: torch.Tensor, context_tokens: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the unified Time-LlaMA model.
        
        Args:
            patch_tokens: [batch_size, seq_len, patch_size] - time series patches
            context_tokens: [batch_size, context_len, context_dim] - GraphRAG context
            context_mask: [batch_size, context_len] - context attention mask
            attention_mask: [batch_size, seq_len] - sequence attention mask
            
        Returns:
            Dictionary containing:
                - forecasts: [batch_size, num_horizons] - forecasting outputs
                - patch_embeddings: [batch_size, seq_len, patch_embed_dim] - patch embeddings
                - hidden_states: [batch_size, seq_len, d_model] - hidden states
        """
        batch_size, seq_len, patch_size = patch_tokens.shape
        
        # Embed patches
        x = self.patch_embedding(patch_tokens)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Apply cross-attention with GraphRAG context if provided
        if context_tokens is not None:
            x = self.cross_attention(x, context_tokens, context_mask)
        
        # Apply transformer blocks
        hidden_states = []
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attention_mask)
            hidden_states.append(x)
        
        # Get final hidden states
        final_hidden = hidden_states[-1]
        
        # Generate outputs
        # Use the last token for forecasting
        last_hidden = final_hidden[:, -1, :]  # [batch_size, d_model]
        forecasts = self.forecast_head(last_hidden)  # [batch_size, num_horizons * output_dim]
        forecasts = forecasts.view(batch_size, self.config.num_horizons, self.config.output_dim)
        forecasts = forecasts.squeeze(-1)  # [batch_size, num_horizons]
        
        # Generate patch embeddings
        patch_embeddings = self.embedding_head(final_hidden)  # [batch_size, seq_len, patch_embed_dim]
        
        return {
            'forecasts': forecasts,
            'patch_embeddings': patch_embeddings,
            'hidden_states': final_hidden
        }
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_unified_time_llama(config: TimeLlaMAConfig, device: str = 'cpu') -> nn.Module:
    """
    Create a unified Time-LlaMA model with optional PEFT/LoRA.
    
    Args:
        config: Time-LlaMA configuration
        device: Device to place the model on
        
    Returns:
        Time-LlaMA model (with PEFT if enabled)
    """
    # Create base model
    model = UnifiedTimeLlaMA(config)
    
    # Apply PEFT/LoRA if enabled (disabled for now due to compatibility issues)
    if False and config.use_peft:
        # Create a simple config dict for PEFT compatibility
        model_config = {
            "tie_word_embeddings": False,
            "vocab_size": 1000,  # Dummy vocab size
            "hidden_size": config.d_model,
            "num_attention_heads": config.n_heads,
            "num_hidden_layers": config.n_layers,
            "intermediate_size": config.d_ff,
            "hidden_dropout_prob": config.dropout,
            "attention_probs_dropout_prob": config.dropout,
        }
        
        # Add the config to the model for PEFT compatibility
        class ModelConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        model.config = ModelConfig(model_config)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Using causal LM for time series
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "ffn.0", "ffn.3"]
        )
        
        model = get_peft_model(model, peft_config)
        logger.info(f"Applied PEFT/LoRA with r={config.lora_r}, alpha={config.lora_alpha}")
    
    # Move to device
    model = model.to(device)
    
    # Log model info
    total_params = model.get_num_parameters()
    trainable_params = model.get_trainable_parameters()
    
    logger.info(f"Created UnifiedTimeLlaMA:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable ratio: {trainable_params/total_params:.2%}")
    
    return model
