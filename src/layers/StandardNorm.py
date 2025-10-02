import torch
import torch.nn as nn


class Normalize(nn.Module):
    """
    Normalization layer for time series data.
    Supports both standard normalization and RevIN (Reversible Instance Normalization).
    
    This module can normalize and denormalize time series data, which is crucial
    for time series forecasting models to handle different scales and distributions.
    """
    def __init__(
        self, 
        num_features: int, 
        eps: float = 1e-5, 
        affine: bool = False, 
        subtract_last: bool = False, 
        non_norm: bool = False
    ):
        """
        Initialize normalization layer.
        
        Args:
            num_features: the number of features or channels
            eps: a value added for numerical stability
            affine: if True, RevIN has learnable affine parameters
            subtract_last: if True, subtract last value instead of mean
            non_norm: if True, skip normalization (identity function)
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Forward pass for normalization/denormalization.
        
        Args:
            x: Input tensor
            mode: 'norm' for normalization, 'denorm' for denormalization
            
        Returns:
            Normalized or denormalized tensor
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")
        return x

    def _init_params(self):
        """Initialize RevIN parameters: (C,)"""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor):
        """Compute normalization statistics."""
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor."""
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize input tensor."""
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x