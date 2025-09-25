import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for time series tokens.
    Adapted from Time-LLM/iTransformer implementations.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class ChannelEmbedding(nn.Module):
    """
    Channel-specific embedding for multivariate time series.
    Each channel gets a learnable embedding vector.
    """
    def __init__(self, num_channels: int, d_model: int):
        super(ChannelEmbedding, self).__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.embedding = nn.Embedding(num_channels, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, d_model] - add channel embeddings
        B, N, d_model = x.shape
        channel_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        channel_emb = self.embedding(channel_ids)
        return x + channel_emb


class TSEmb(nn.Module):
    """
    Time-LlaMA Channel-as-token embedding with enhanced features.
    
    Input:  x of shape [batch_size, lookback_steps, num_channels]
    Output: tokens of shape [batch_size, num_channels, d_model]
    
    Implements the paper's linear tokenization with optional enhancements:
    - Linear projection of each channel's entire window
    - Optional positional encoding for temporal information
    - Optional channel-specific embeddings
    - Dropout for regularization
    """
    
    def __init__(
        self, 
        lookback: int, 
        d_model: int, 
        num_channels: Optional[int] = None,
        use_positional_emb: bool = True,
        use_channel_emb: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.lookback = lookback
        self.d_model = d_model
        self.num_channels = num_channels
        self.use_positional_emb = use_positional_emb
        self.use_channel_emb = use_channel_emb
        
        # Main projection: maps lookback steps to d_model
        self.proj = nn.Linear(lookback, d_model)
        
        # Optional positional encoding for temporal information
        if use_positional_emb:
            self.pos_embedding = PositionalEmbedding(d_model, max_len=lookback)
        
        # Optional channel-specific embeddings
        if use_channel_emb and num_channels is not None:
            self.channel_embedding = ChannelEmbedding(num_channels, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for channel-as-token embedding.
        
        Args:
            x: Input tensor of shape [B, TL, N] where
               B = batch size, TL = lookback steps, N = num channels
        
        Returns:
            tokens: Output tensor of shape [B, N, d_model]
        """
        # x: [B, TL, N]
        assert x.dim() == 3, "Expected input of shape [B, TL, N]"
        B, TL, N = x.shape
        assert TL == self.lookback, f"lookback mismatch: got {TL}, expected {self.lookback}"
        
        # Reorder to [B, N, TL] so Linear maps TL -> d_model per channel
        x = x.transpose(1, 2)  # [B, N, TL]
        tokens = self.proj(x)  # [B, N, d_model]
        
        # Add positional encoding if enabled
        if self.use_positional_emb:
            pos_emb = self.pos_embedding(tokens)  # [1, N, d_model]
            tokens = tokens + pos_emb
        
        # Add channel-specific embeddings if enabled
        if self.use_channel_emb and hasattr(self, 'channel_embedding'):
            tokens = self.channel_embedding(tokens)
        
        # Apply dropout
        tokens = self.dropout(tokens)
        
        return tokens


class TSEmbConv(nn.Module):
    """
    Alternative channel-as-token embedding using 1D convolution.
    Inspired by Time-LLM's TokenEmbedding but adapted for channel-as-token.
    """
    
    def __init__(
        self, 
        lookback: int, 
        d_model: int, 
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.lookback = lookback
        self.d_model = d_model
        
        # Use 1D convolution for temporal feature extraction
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=lookback, 
            out_channels=d_model,
            kernel_size=kernel_size, 
            padding=padding, 
            padding_mode='circular', 
            bias=False
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize conv weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu'
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using 1D convolution.
        
        Args:
            x: Input tensor of shape [B, TL, N]
        
        Returns:
            tokens: Output tensor of shape [B, N, d_model]
        """
        # x: [B, TL, N]
        assert x.dim() == 3, "Expected input of shape [B, TL, N]"
        B, TL, N = x.shape
        assert TL == self.lookback, f"lookback mismatch: got {TL}, expected {self.lookback}"
        
        # Reorder to [B, N, TL] and apply conv1d
        x = x.transpose(1, 2)  # [B, N, TL]
        x = x.permute(0, 2, 1)  # [B, TL, N] for conv1d
        tokens = self.conv(x)  # [B, d_model, N]
        tokens = tokens.permute(0, 2, 1)  # [B, N, d_model]
        
        return self.dropout(tokens)


class TSEmbHybrid(nn.Module):
    """
    Hybrid embedding combining linear projection and convolution.
    Uses both approaches and concatenates or adds their outputs.
    """
    
    def __init__(
        self, 
        lookback: int, 
        d_model: int, 
        num_channels: Optional[int] = None,
        use_conv: bool = True,
        use_linear: bool = True,
        fusion_method: str = 'add',  # 'add', 'concat', 'gate'
        dropout: float = 0.1
    ):
        super().__init__()
        self.lookback = lookback
        self.d_model = d_model
        self.fusion_method = fusion_method
        
        if use_linear:
            self.linear_proj = nn.Linear(lookback, d_model)
        
        if use_conv:
            self.conv_proj = TSEmbConv(lookback, d_model, dropout=0.0)
        
        if fusion_method == 'concat':
            # If concatenating, we need to project back to d_model
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        elif fusion_method == 'gate':
            # Gating mechanism to weight between linear and conv
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hybrid embedding.
        """
        # x: [B, TL, N]
        assert x.dim() == 3, "Expected input of shape [B, TL, N]"
        B, TL, N = x.shape
        
        outputs = []
        
        if hasattr(self, 'linear_proj'):
            # Linear projection
            x_linear = x.transpose(1, 2)  # [B, N, TL]
            linear_out = self.linear_proj(x_linear)  # [B, N, d_model]
            outputs.append(linear_out)
        
        if hasattr(self, 'conv_proj'):
            # Convolutional projection
            conv_out = self.conv_proj(x)  # [B, N, d_model]
            outputs.append(conv_out)
        
        if len(outputs) == 1:
            tokens = outputs[0]
        elif self.fusion_method == 'add':
            tokens = outputs[0] + outputs[1]
        elif self.fusion_method == 'concat':
            tokens = torch.cat(outputs, dim=-1)  # [B, N, 2*d_model]
            tokens = self.fusion_proj(tokens)  # [B, N, d_model]
        elif self.fusion_method == 'gate':
            concat_tokens = torch.cat(outputs, dim=-1)  # [B, N, 2*d_model]
            gate_weights = self.gate(concat_tokens)  # [B, N, d_model]
            tokens = gate_weights * outputs[0] + (1 - gate_weights) * outputs[1]
        
        return self.dropout(tokens)
