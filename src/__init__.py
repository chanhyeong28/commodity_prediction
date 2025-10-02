"""Time-LlaMA implementation with Time-LLM inspired features.

This package contains a comprehensive implementation of Time-LlaMA that combines:
- Channel-as-token embedding (Time-LlaMA's core innovation)
- Prompt alignment via cross-attention
- Reprogramming layer for vocabulary alignment (from Time-LLM)
- LLM backbone usage with frozen parameters
- Multi-scale output projection
- Normalization and preprocessing utilities

The implementation follows Time-LLM's architecture patterns while implementing
Time-LlaMA's channel-as-token approach for efficient time series forecasting.
"""

# Import main model
from .models import TimeLlaMA, FlattenHead, ReprogrammingLayer

# Import key layers
from .layers import (
    TSEmb, TSEmbConv, TSEmbHybrid,
    PromptAlignment, MultiHeadAttention, CrossAttention,
    TimeLlaMAEncoder, TimeLlaMADecoder, Normalize
)

__version__ = "0.1.0"

__all__ = [
    # Main model
    "TimeLlaMA",
    "FlattenHead", 
    "ReprogrammingLayer",
    
    # Embeddings
    "TSEmb",
    "TSEmbConv", 
    "TSEmbHybrid",
    
    # Attention mechanisms
    "PromptAlignment",
    "MultiHeadAttention", 
    "CrossAttention",
    
    # Transformer components
    "TimeLlaMAEncoder",
    "TimeLlaMADecoder",
    
    # Utilities
    "Normalize",
]


