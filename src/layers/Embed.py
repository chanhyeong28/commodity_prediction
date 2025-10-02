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


# ============================================================================
# Time-LLM Components (from references/Time-LLM/layers/Embed.py)
# ============================================================================

class ReplicationPad1d(nn.Module):
    """
    Replication padding for 1D tensors.
    Used by PatchEmbedding for proper patch extraction.
    """
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output


class TokenEmbedding(nn.Module):
    """
    Token embedding using 1D convolution.
    Used for converting time series patches to embeddings.
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    Fixed sinusoidal embedding for categorical features.
    """
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    Temporal embedding for time features (hour, day, month, etc.).
    """
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    Time feature embedding using linear projection.
    """
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    Comprehensive data embedding combining value, position, and temporal embeddings.
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x).to(x.device)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """
    Data embedding without positional encoding.
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Patch-based embedding for Time-LLM.
    Converts time series into patches and embeds them.
    """
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


class DataEmbedding_wo_time(nn.Module):
    """
    Data embedding without temporal features.
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_time, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
