import torch
import torch.nn as nn
from .SelfAttention_Family import CrossAttention


class PromptAlignment(nn.Module):
    """
    Multi-head cross-attention alignment module for Time-LlaMA.
    
    This module aligns time series tokens with text prompt embeddings using
    cross-attention mechanism. It's a key component for bridging the gap
    between time series and natural language modalities.

    Args:
        d_model: Model dimension (for time series tokens)
        d_llm: LLM dimension (for prompt embeddings)
        num_heads: Number of attention heads
        attention_dropout: Dropout rate for attention weights
        use_residual: Whether to use residual connection
    """

    def __init__(
        self, 
        d_model: int, 
        d_llm: int,
        num_heads: int, 
        attention_dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.d_llm = d_llm
        self.num_heads = num_heads
        self.use_residual = use_residual
        
        # Project prompt embeddings from d_llm to d_model for cross-attention
        self.prompt_projection = nn.Linear(d_llm, d_model)
        
        # Use the new CrossAttention implementation
        self.cross_attention = CrossAttention(
            d_model=d_model,
            n_heads=num_heads,
            attention_dropout=attention_dropout
        )
        
        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Optional feed-forward network for additional processing
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(attention_dropout)
        )

    def forward(self, H0: torch.Tensor, HP: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for prompt alignment.
        
        Args:
            H0: Time series tokens [B, N, d_model]
            HP: Prompt embeddings [B, P, d_llm]
            
        Returns:
            aligned_tokens: Aligned time series tokens [B, N, d_model]
        """
        # H0: [B, N, d_model], HP: [B, P, d_llm]
        B, N, D = H0.shape
        _, P, Dp = HP.shape
        assert D == self.d_model, f"Time series dimension mismatch: H0={D}, expected={self.d_model}"
        assert Dp == self.d_llm, f"Prompt dimension mismatch: HP={Dp}, expected={self.d_llm}"

        # Project prompt embeddings from d_llm to d_model
        HP_projected = self.prompt_projection(HP)  # [B, P, d_model]

        # Apply cross-attention
        aligned_tokens = self.cross_attention(H0, HP_projected, HP_projected)
        
        # Apply residual connection if enabled
        if self.use_residual:
            aligned_tokens = H0 + aligned_tokens
        
        # Apply layer normalization
        aligned_tokens = self.layer_norm(aligned_tokens)
        
        # Optional feed-forward processing
        ffn_out = self.ffn(aligned_tokens)
        aligned_tokens = aligned_tokens + ffn_out
        
        return aligned_tokens


class PromptAlignmentSimple(nn.Module):
    """
    Simplified prompt alignment module (original implementation).
    Kept for backward compatibility and comparison.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, H0: torch.Tensor, HP: torch.Tensor) -> torch.Tensor:
        # H0: [B, N, d_model], HP: [B, P, d_model]
        B, N, D = H0.shape
        _, P, Dp = HP.shape
        assert D == self.d_model and Dp == self.d_model

        Q = self.Wq(H0).view(B, N, self.num_heads, self.d_head).transpose(1, 2)  # [B, h, N, d]
        K = self.Wk(HP).view(B, P, self.num_heads, self.d_head).transpose(1, 2)  # [B, h, P, d]
        V = self.Wv(HP).view(B, P, self.num_heads, self.d_head).transpose(1, 2)  # [B, h, P, d]

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B, h, N, P]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, V)  # [B, h, N, d]
        context = context.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = H0 + self.Wo(context)  # residual per paper Eq (3)
        return out


