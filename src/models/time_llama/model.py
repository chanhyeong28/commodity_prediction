"""
Simplified Time-LlaMA Implementation

This module provides a simplified but working Time-LlaMA implementation
that focuses on the core functionality without complex dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SimpleTimeLlaMAConfig:
    """Simplified configuration for Time-LlaMA model"""
    def __init__(
        self,
        patch_size: int = 7,
        patch_stride: int = 1,
        patch_embed_dim: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd_per_head: int = 64,
        dropout: float = 0.0,
        context_dim: int = 64,
        context_heads: int = 2,
        max_context_tokens: int = 256,
        prediction_length: int = 4,
    ):
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_embed_dim = patch_embed_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd_per_head = n_embd_per_head
        self.dropout = dropout
        self.context_dim = context_dim
        self.context_heads = context_heads
        self.max_context_tokens = max_context_tokens
        self.prediction_length = prediction_length
        
        # Computed properties
        self.n_embd = n_embd_per_head * n_head


class PatchEmbedding(nn.Module):
    """Simple patch embedding using 1D CNN"""
    
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Simple 1D CNN for patch embedding
        self.conv1 = nn.Conv1d(1, embed_dim // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: [batch_size, num_patches, patch_size]
        Returns:
            embeddings: [batch_size, num_patches, embed_dim]
        """
        batch_size, num_patches, patch_len = patches.shape
        
        # Reshape for convolution: [batch_size * num_patches, 1, patch_size]
        x = patches.view(batch_size * num_patches, 1, patch_len)
        
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Global average pooling
        x = self.pool(x)  # [batch_size * num_patches, embed_dim, 1]
        x = x.squeeze(-1)  # [batch_size * num_patches, embed_dim]
        
        # Reshape back to patch format
        x = x.view(batch_size, num_patches, self.embed_dim)
        
        # Normalize and dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class CrossAttentionAdapter(nn.Module):
    """Cross-attention adapter for GraphRAG context integration"""
    
    def __init__(self, d_model: int, context_dim: int, n_heads: int = 2):
        super().__init__()
        self.d_model = d_model
        self.context_dim = context_dim
        self.n_heads = n_heads
        
        # Project context to match model dimension
        self.context_proj = nn.Linear(context_dim, d_model)
        
        # Cross-attention layers
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        
    def forward(self, time_tokens: torch.Tensor, context_tokens: torch.Tensor, 
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            time_tokens: [batch_size, seq_len, d_model] - Time series patch tokens
            context_tokens: [batch_size, context_len, context_dim] - GraphRAG context
            context_mask: [batch_size, context_len] - Context attention mask
        Returns:
            enhanced_tokens: [batch_size, seq_len, d_model] - Enhanced time tokens
        """
        # Project context to model dimension
        context_proj = self.context_proj(context_tokens)  # [batch_size, context_len, d_model]
        
        # Cross-attention: time tokens attend to context
        attn_output, _ = self.cross_attn(
            query=time_tokens,
            key=context_proj,
            value=context_proj,
            key_padding_mask=context_mask
        )
        
        # Residual connection and normalization
        time_tokens = self.norm1(time_tokens + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(time_tokens)
        time_tokens = self.norm2(time_tokens + ffn_output)
        
        return time_tokens


class TransformerBlock(nn.Module):
    """Simple transformer block with self-attention"""
    
    def __init__(self, config: SimpleTimeLlaMAConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.ReLU(),
            nn.Linear(config.n_embd * 4, config.n_embd),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class SimpleTimeLlaMA(nn.Module):
    """
    Simplified Time-LlaMA model with Lag-Llama-inspired architecture.
    
    This model processes time series patches and integrates GraphRAG context
    through cross-attention, following the Time-LlaMA strategy.
    """
    
    def __init__(self, config: SimpleTimeLlaMAConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_size=config.patch_size,
            embed_dim=config.patch_embed_dim
        )
        
        # Project patch embeddings to model dimension
        self.patch_proj = nn.Linear(config.patch_embed_dim, config.n_embd)
        
        # Cross-attention adapter for GraphRAG context
        self.cross_attention = CrossAttentionAdapter(
            d_model=config.n_embd,
            context_dim=config.context_dim,
            n_heads=config.context_heads
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output head for forecasting
        self.output_head = nn.Linear(
            config.n_embd,
            config.prediction_length
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, patches: torch.Tensor, context_tokens: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Time-LlaMA model.
        
        Args:
            patches: [batch_size, num_patches, patch_size] - Time series patches
            context_tokens: [batch_size, context_len, context_dim] - GraphRAG context (optional)
            context_mask: [batch_size, context_len] - Context attention mask (optional)
            attention_mask: [batch_size, num_patches] - Patch attention mask (optional)
            
        Returns:
            Dictionary containing forecasts and embeddings
        """
        batch_size, num_patches, patch_size = patches.shape
        
        # Embed patches
        patch_embeddings = self.patch_embedding(patches)  # [batch_size, num_patches, patch_embed_dim]
        
        # Project to model dimension
        x = self.patch_proj(patch_embeddings)  # [batch_size, num_patches, n_embd]
        
        # Apply cross-attention with GraphRAG context if provided
        if context_tokens is not None:
            x = self.cross_attention(x, context_tokens, context_mask)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Global average pooling over patches (with attention mask if provided)
        if attention_mask is not None:
            # Apply attention mask
            x = x * attention_mask.unsqueeze(-1)
            # Weighted average
            pooled = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = x.mean(dim=1)  # [batch_size, n_embd]
        
        # Generate forecasts
        forecasts = self.output_head(pooled)  # [batch_size, prediction_length]
        
        return {
            'forecasts': forecasts,
            'patch_embeddings': patch_embeddings,
            'hidden_states': x
        }
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleTimeLlaMAWithPEFT(nn.Module):
    """
    Simple Time-LlaMA model with basic PEFT-style parameter efficiency.
    
    This version uses a simpler approach to parameter efficiency without
    complex PEFT dependencies.
    """
    
    def __init__(self, config: SimpleTimeLlaMAConfig, use_peft: bool = True):
        super().__init__()
        self.config = config
        self.use_peft = use_peft
        
        # Create the backbone model
        self.backbone = SimpleTimeLlaMA(config)
        
        # Apply simple parameter efficiency if requested
        if use_peft:
            self._apply_simple_peft()
        
        logger.info(f"Simple Time-LlaMA model created with {self.get_num_parameters():,} parameters")
        if use_peft:
            logger.info(f"Trainable parameters: {self.get_trainable_parameters():,}")
    
    def _apply_simple_peft(self):
        """Apply simple parameter efficiency by freezing some layers"""
        # Freeze patch embedding and projection layers
        for param in self.backbone.patch_embedding.parameters():
            param.requires_grad = False
        for param in self.backbone.patch_proj.parameters():
            param.requires_grad = False
        
        # Keep cross-attention and transformer blocks trainable
        # This is a simple form of parameter efficiency
    
    def forward(self, patches: torch.Tensor, context_tokens: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        return self.backbone(patches, context_tokens, context_mask, attention_mask)
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_simple_time_llama(
    patch_size: int = 7,
    patch_stride: int = 1,
    patch_embed_dim: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd_per_head: int = 64,
    dropout: float = 0.0,
    context_dim: int = 64,
    context_heads: int = 2,
    max_context_tokens: int = 256,
    prediction_length: int = 4,
    use_peft: bool = True,
) -> SimpleTimeLlaMAWithPEFT:
    """
    Convenience function to create a simple Time-LlaMA model.
    
    Args:
        patch_size: Size of time series patches
        patch_stride: Stride between patches
        patch_embed_dim: Embedding dimension for patches
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd_per_head: Embedding dimension per head
        dropout: Dropout rate
        context_dim: Dimension for GraphRAG context
        context_heads: Number of heads for cross-attention
        max_context_tokens: Maximum context tokens
        prediction_length: Prediction horizon
        use_peft: Whether to use parameter efficiency
        
    Returns:
        Simple Time-LlaMA model
    """
    config = SimpleTimeLlaMAConfig(
        patch_size=patch_size,
        patch_stride=patch_stride,
        patch_embed_dim=patch_embed_dim,
        n_layer=n_layer,
        n_head=n_head,
        n_embd_per_head=n_embd_per_head,
        dropout=dropout,
        context_dim=context_dim,
        context_heads=context_heads,
        max_context_tokens=max_context_tokens,
        prediction_length=prediction_length,
    )
    
    return SimpleTimeLlaMAWithPEFT(config, use_peft)
