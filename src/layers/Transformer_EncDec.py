import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from .SelfAttention_Family import MultiHeadAttention, CrossAttention


class ConvLayer(nn.Module):
    """
    Convolutional layer for downsampling and feature extraction.
    Used in encoder layers for hierarchical feature learning.
    """
    def __init__(
        self, 
        c_in: int, 
        kernel_size: int = 3, 
        stride: int = 2, 
        padding: int = 1,
        activation: str = "elu"
    ):
        super(ConvLayer, self).__init__()
        self.c_in = c_in
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Downsampling convolution
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='circular'
        )
        
        # Normalization
        self.norm = nn.BatchNorm1d(c_in)
        
        # Activation function
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ELU()
        
        # Max pooling for downsampling
        self.maxPool = nn.MaxPool1d(
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize convolution weights."""
        nn.init.kaiming_normal_(self.downConv.weight, mode='fan_in', nonlinearity='relu')
        if self.downConv.bias is not None:
            nn.init.zeros_(self.downConv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for convolutional layer.
        
        Args:
            x: Input tensor of shape [B, L, C]
            
        Returns:
            output: Downsampled tensor of shape [B, L', C]
        """
        # x: [B, L, C] -> [B, C, L]
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        # [B, C, L'] -> [B, L', C]
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """
    Transformer encoder layer with self-attention and feed-forward network.
    Enhanced for Time-LlaMA with configurable attention mechanisms.
    """
    def __init__(
        self, 
        attention: nn.Module,
        d_model: int, 
        d_ff: Optional[int] = None, 
        dropout: float = 0.1, 
        activation: str = "relu",
        use_conv_ffn: bool = True,
        pre_norm: bool = False
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = attention
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_conv_ffn = use_conv_ffn
        self.pre_norm = pre_norm
        
        # Feed-forward network
        if use_conv_ffn:
            # Convolutional FFN (original approach)
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        else:
            # Standard linear FFN
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "elu":
            self.activation = F.elu
        else:
            self.activation = F.relu
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize feed-forward weights."""
        if self.use_conv_ffn:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
            if self.conv1.bias is not None:
                nn.init.zeros_(self.conv1.bias)
            if self.conv2.bias is not None:
                nn.init.zeros_(self.conv2.bias)
        else:
            for module in self.ffn:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None, 
        tau: Optional[torch.Tensor] = None, 
        delta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for encoder layer.
        
        Args:
            x: Input tensor of shape [B, L, D]
            attn_mask: Optional attention mask
            tau: Optional de-stationary factor
            delta: Optional de-stationary factor
            
        Returns:
            output: Encoded tensor of shape [B, L, D]
            attention_weights: Optional attention weights
        """
        # Self-attention with residual connection
        if self.pre_norm:
            # Pre-norm architecture
            norm_x = self.norm1(x)
            new_x, attn = self.attention(
                norm_x, norm_x, norm_x,
                attn_mask=attn_mask,
                tau=tau, delta=delta
            )
            x = x + self.dropout(new_x)
        else:
            # Post-norm architecture (original)
            new_x, attn = self.attention(
                x, x, x,
                attn_mask=attn_mask,
                tau=tau, delta=delta
            )
            x = x + self.dropout(new_x)
            x = self.norm1(x)

        # Feed-forward network with residual connection
        if self.use_conv_ffn:
            # Convolutional FFN
            y = x
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
        else:
            # Linear FFN
            y = self.ffn(x)
        
        if self.pre_norm:
            x = x + y
            x = self.norm2(x)
        else:
            x = self.norm2(x + y)

        return x, attn


class Encoder(nn.Module):
    """
    Transformer encoder with multiple encoder layers.
    Supports both standard and hierarchical (with conv layers) architectures.
    """
    def __init__(
        self, 
        attn_layers: List[nn.Module], 
        conv_layers: Optional[List[nn.Module]] = None, 
        norm_layer: Optional[nn.Module] = None,
        output_attention: bool = False
    ):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.output_attention = output_attention

    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None, 
        tau: Optional[torch.Tensor] = None, 
        delta: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass for encoder.
        
        Args:
            x: Input tensor of shape [B, L, D]
            attn_mask: Optional attention mask
            tau: Optional de-stationary factor
            delta: Optional de-stationary factor
            
        Returns:
            output: Encoded tensor of shape [B, L', D]
            attention_weights: List of attention weights (if output_attention=True)
        """
        attns = []
        
        if self.conv_layers is not None:
            # Hierarchical encoder with convolutional layers
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                # Apply de-stationary factor only to the first layer
                layer_delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=layer_delta)
                x = conv_layer(x)
                if self.output_attention:
                    attns.append(attn)
            
            # Final attention layer without convolution
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            if self.output_attention:
                attns.append(attn)
        else:
            # Standard encoder without convolutional layers
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                if self.output_attention:
                    attns.append(attn)

        # Final layer normalization
        if self.norm is not None:
            x = self.norm(x)

        if self.output_attention:
            return x, attns
        else:
            return x


class DecoderLayer(nn.Module):
    """
    Transformer decoder layer with self-attention, cross-attention, and feed-forward network.
    Enhanced for Time-LlaMA with configurable attention mechanisms.
    """
    def __init__(
        self, 
        self_attention: nn.Module, 
        cross_attention: nn.Module, 
        d_model: int, 
        d_ff: Optional[int] = None,
        dropout: float = 0.1, 
        activation: str = "relu",
        use_conv_ffn: bool = True,
        pre_norm: bool = False
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_conv_ffn = use_conv_ffn
        self.pre_norm = pre_norm
        
        # Feed-forward network
        if use_conv_ffn:
            # Convolutional FFN
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        else:
            # Standard linear FFN
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "elu":
            self.activation = F.elu
        else:
            self.activation = F.relu
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize feed-forward weights."""
        if self.use_conv_ffn:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
            if self.conv1.bias is not None:
                nn.init.zeros_(self.conv1.bias)
            if self.conv2.bias is not None:
                nn.init.zeros_(self.conv2.bias)
        else:
            for module in self.ffn:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self, 
        x: torch.Tensor, 
        cross: torch.Tensor, 
        x_mask: Optional[torch.Tensor] = None, 
        cross_mask: Optional[torch.Tensor] = None, 
        tau: Optional[torch.Tensor] = None, 
        delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for decoder layer.
        
        Args:
            x: Target tensor of shape [B, L_tgt, D]
            cross: Source tensor of shape [B, L_src, D]
            x_mask: Optional target attention mask
            cross_mask: Optional source attention mask
            tau: Optional de-stationary factor
            delta: Optional de-stationary factor
            
        Returns:
            output: Decoded tensor of shape [B, L_tgt, D]
        """
        # Self-attention
        if self.pre_norm:
            norm_x = self.norm1(x)
            x = x + self.dropout(self.self_attention(
                norm_x, norm_x, norm_x,
                attn_mask=x_mask,
                tau=tau, delta=None
            )[0])
        else:
            x = x + self.dropout(self.self_attention(
                x, x, x,
                attn_mask=x_mask,
                tau=tau, delta=None
            )[0])
            x = self.norm1(x)

        # Cross-attention
        if self.pre_norm:
            norm_x = self.norm2(x)
            x = x + self.dropout(self.cross_attention(
                norm_x, cross, cross,
                attn_mask=cross_mask,
                tau=tau, delta=delta
            )[0])
        else:
            x = x + self.dropout(self.cross_attention(
                x, cross, cross,
                attn_mask=cross_mask,
                tau=tau, delta=delta
            )[0])
            x = self.norm2(x)

        # Feed-forward network
        if self.use_conv_ffn:
            # Convolutional FFN
            y = x
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
        else:
            # Linear FFN
            y = self.ffn(x)
        
        if self.pre_norm:
            x = x + y
            x = self.norm3(x)
        else:
            x = self.norm3(x + y)

        return x


class Decoder(nn.Module):
    """
    Transformer decoder with multiple decoder layers.
    Enhanced for Time-LlaMA with optional projection layer.
    """
    def __init__(
        self, 
        layers: List[nn.Module], 
        norm_layer: Optional[nn.Module] = None, 
        projection: Optional[nn.Module] = None,
        output_attention: bool = False
    ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        self.output_attention = output_attention

    def forward(
        self, 
        x: torch.Tensor, 
        cross: torch.Tensor, 
        x_mask: Optional[torch.Tensor] = None, 
        cross_mask: Optional[torch.Tensor] = None, 
        tau: Optional[torch.Tensor] = None, 
        delta: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass for decoder.
        
        Args:
            x: Target tensor of shape [B, L_tgt, D]
            cross: Source tensor of shape [B, L_src, D]
            x_mask: Optional target attention mask
            cross_mask: Optional source attention mask
            tau: Optional de-stationary factor
            delta: Optional de-stationary factor
            
        Returns:
            output: Decoded tensor of shape [B, L_tgt, D']
            attention_weights: List of attention weights (if output_attention=True)
        """
        attns = []
        
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
            if self.output_attention:
                # Note: Individual layer attention weights would need to be collected
                # This is a simplified version
                attns.append(None)

        # Final layer normalization
        if self.norm is not None:
            x = self.norm(x)

        # Optional projection layer
        if self.projection is not None:
            x = self.projection(x)
        
        if self.output_attention:
            return x, attns
        else:
            return x


class TimeLlaMAEncoder(nn.Module):
    """
    Time-LlaMA specific encoder that combines channel-as-token processing
    with transformer encoder architecture.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        attention_type: str = "full",
        use_conv_layers: bool = False,
        pre_norm: bool = False
    ):
        super(TimeLlaMAEncoder, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # Create attention layers
        attn_layers = []
        for _ in range(n_layers):
            attention = MultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                attention_type=attention_type,
                attention_dropout=dropout
            )
            attn_layers.append(EncoderLayer(
                attention=attention,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm
            ))
        
        # Create convolutional layers if needed
        conv_layers = None
        if use_conv_layers:
            conv_layers = [ConvLayer(d_model) for _ in range(n_layers - 1)]
        
        # Create encoder
        self.encoder = Encoder(
            attn_layers=attn_layers,
            conv_layers=conv_layers,
            norm_layer=nn.LayerNorm(d_model) if pre_norm else None,
            output_attention=False
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for Time-LlaMA encoder.
        
        Args:
            x: Input tensor of shape [B, N, D] (channel-as-token)
            attn_mask: Optional attention mask
            
        Returns:
            output: Encoded tensor of shape [B, N, D]
        """
        return self.encoder(x, attn_mask=attn_mask)


class TimeLlaMADecoder(nn.Module):
    """
    Time-LlaMA specific decoder for autoregressive generation.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        attention_type: str = "full",
        pre_norm: bool = False
    ):
        super(TimeLlaMADecoder, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # Create decoder layers
        layers = []
        for _ in range(n_layers):
            self_attention = MultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                attention_type=attention_type,
                attention_dropout=dropout
            )
            cross_attention = CrossAttention(
                d_model=d_model,
                n_heads=n_heads,
                attention_dropout=dropout
            )
            layers.append(DecoderLayer(
                self_attention=self_attention,
                cross_attention=cross_attention,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm
            ))
        
        # Create decoder
        self.decoder = Decoder(
            layers=layers,
            norm_layer=nn.LayerNorm(d_model) if pre_norm else None,
            output_attention=False
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        cross: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for Time-LlaMA decoder.
        
        Args:
            x: Target tensor of shape [B, L_tgt, D]
            cross: Source tensor of shape [B, L_src, D]
            x_mask: Optional target attention mask
            cross_mask: Optional source attention mask
            
        Returns:
            output: Decoded tensor of shape [B, L_tgt, D]
        """
        return self.decoder(x, cross, x_mask=x_mask, cross_mask=cross_mask)
