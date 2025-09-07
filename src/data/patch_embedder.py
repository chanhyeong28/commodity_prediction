"""
Patch-based Embedding System for Time Series Windows

This module implements the patch-as-token approach where each time series window
becomes a token in the Time-LlaMA-style model. It handles multiple window sizes,
overlapping patches, and creates embeddings for all series in the same space.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

@dataclass
class PatchConfig:
    """Configuration for patch generation and embedding"""
    window_sizes: List[int] = None  # [7, 14, 28] days
    strides: List[int] = None        # [1, 3, 7] days overlap
    embedding_dim: int = 128
    max_patches_per_series: int = 100
    normalize_patches: bool = True
    
    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [7, 14, 28]
        if self.strides is None:
            self.strides = [1, 3, 7]

class PatchEmbedder(nn.Module):
    """
    Neural network for embedding time series patches.
    
    This module converts time series windows into dense embeddings that can be
    used as tokens in the Time-LlaMA model. It supports multiple architectures:
    - 1D CNN with pooling
    - Small Transformer encoder
    - Hybrid approach
    """
    
    def __init__(self, 
                 input_dim: int = 1,
                 embedding_dim: int = 128,
                 max_window_size: int = 28,
                 architecture: str = 'cnn',
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.max_window_size = max_window_size
        self.architecture = architecture
        
        if architecture == 'cnn':
            self._build_cnn_encoder()
        elif architecture == 'transformer':
            self._build_transformer_encoder()
        elif architecture == 'hybrid':
            self._build_hybrid_encoder()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
    def _build_cnn_encoder(self):
        """Build 1D CNN encoder"""
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.input_dim, 32, kernel_size=3, padding=1),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, self.embedding_dim)
        
    def _build_transformer_encoder(self):
        """Build small Transformer encoder"""
        self.pos_embedding = nn.Parameter(torch.randn(self.max_window_size, self.embedding_dim))
        self.input_projection = nn.Linear(self.input_dim, self.embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def _build_hybrid_encoder(self):
        """Build hybrid CNN + Transformer encoder"""
        # CNN for local features
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.input_dim, 32, kernel_size=3, padding=1),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
        ])
        
        # Transformer for global context
        self.input_projection = nn.Linear(64, self.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding.
        
        Args:
            patches: Tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Embeddings of shape [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, input_dim = patches.shape
        
        if self.architecture == 'cnn':
            return self._forward_cnn(patches)
        elif self.architecture == 'transformer':
            return self._forward_transformer(patches)
        elif self.architecture == 'hybrid':
            return self._forward_hybrid(patches)
    
    def _forward_cnn(self, patches: torch.Tensor) -> torch.Tensor:
        """CNN forward pass"""
        # Reshape for conv1d: [batch_size, input_dim, seq_len]
        x = patches.transpose(1, 2)
        
        # Apply conv layers
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        # Global pooling
        x = self.pool(x)  # [batch_size, 128, 1]
        x = x.squeeze(-1)  # [batch_size, 128]
        
        # Project to embedding dimension
        x = self.fc(x)  # [batch_size, embedding_dim]
        
        # Expand to sequence length
        x = x.unsqueeze(1).expand(-1, patches.size(1), -1)
        
        return self.dropout(x)
    
    def _forward_transformer(self, patches: torch.Tensor) -> torch.Tensor:
        """Transformer forward pass"""
        batch_size, seq_len, _ = patches.shape
        
        # Project input
        x = self.input_projection(patches)  # [batch_size, seq_len, embedding_dim]
        
        # Add positional embedding
        x = x + self.pos_embedding[:seq_len].unsqueeze(0)
        
        # Apply transformer
        x = self.transformer(x)
        
        return self.dropout(x)
    
    def _forward_hybrid(self, patches: torch.Tensor) -> torch.Tensor:
        """Hybrid CNN + Transformer forward pass"""
        batch_size, seq_len, input_dim = patches.shape
        
        # CNN for local features
        x = patches.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        # Transpose back for transformer
        x = x.transpose(1, 2)  # [batch_size, seq_len, 64]
        
        # Project to embedding dimension
        x = self.input_projection(x)  # [batch_size, seq_len, embedding_dim]
        
        # Apply transformer
        x = self.transformer(x)
        
        return self.dropout(x)

class PatchTokenizer:
    """
    Converts time series data into patches that can be embedded.
    
    This class handles:
    1. Creating overlapping patches with different window sizes
    2. Normalizing patches
    3. Managing patch sequences for multiple series
    """
    
    def __init__(self, config: PatchConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame):
        """
        Fit the tokenizer on training data.
        
        Args:
            data: Training dataframe with time series columns
        """
        # Collect all values for normalization
        all_values = []
        for column in data.columns:
            if column != 'date_id':
                series_data = data[column].dropna()
                all_values.extend(series_data.values)
        
        # Fit scaler
        self.scaler.fit(np.array(all_values).reshape(-1, 1))
        self.is_fitted = True
        
        logger.info(f"Fitted tokenizer on {len(all_values)} values")
    
    def create_patches(self, series_data: pd.Series, series_id: str) -> List[Dict[str, Any]]:
        """
        Create patches from a time series.
        
        Args:
            series_data: Time series data
            series_id: Series identifier
            
        Returns:
            List of patch dictionaries
        """
        patches = []
        
        for window_size in self.config.window_sizes:
            for stride in self.config.strides:
                # Create overlapping windows
                for start_idx in range(0, len(series_data) - window_size + 1, stride):
                    end_idx = start_idx + window_size
                    window_data = series_data.iloc[start_idx:end_idx]
                    
                    # Skip if insufficient data
                    if len(window_data) < window_size * 0.8:  # Allow some missing data
                        continue
                    
                    # Normalize if configured
                    if self.config.normalize_patches and self.is_fitted:
                        window_values = self.scaler.transform(window_data.values.reshape(-1, 1)).flatten()
                    else:
                        window_values = window_data.values
                    
                    patch = {
                        'series_id': series_id,
                        'window_size': window_size,
                        'stride': stride,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'start_date': series_data.index[start_idx],
                        'end_date': series_data.index[end_idx - 1],
                        'values': window_values,
                        'patch_id': f"{series_id}_{window_size}d_{stride}s_{start_idx}_{end_idx}"
                    }
                    
                    patches.append(patch)
        
        # Limit patches per series
        if len(patches) > self.config.max_patches_per_series:
            # Keep most recent patches
            patches.sort(key=lambda x: x['end_date'])
            patches = patches[-self.config.max_patches_per_series:]
        
        return patches
    
    def create_patch_sequences(self, data: pd.DataFrame, 
                             target_series: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create patch sequences for all series in the dataframe.
        
        Args:
            data: Dataframe with time series
            target_series: List of target series (if None, use all)
            
        Returns:
            Dictionary mapping series_id to list of patches
        """
        if target_series is None:
            target_series = [col for col in data.columns if col != 'date_id']
        
        patch_sequences = {}
        
        for series_id in target_series:
            if series_id not in data.columns:
                logger.warning(f"Series {series_id} not found in data")
                continue
            
            series_data = data[series_id].dropna()
            if len(series_data) < min(self.config.window_sizes):
                logger.warning(f"Insufficient data for {series_id}")
                continue
            
            patches = self.create_patches(series_data, series_id)
            patch_sequences[series_id] = patches
            
            logger.info(f"Created {len(patches)} patches for {series_id}")
        
        return patch_sequences
    
    def pad_patches(self, patches: List[Dict[str, Any]], 
                   max_window_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad patches to the same length for batch processing.
        
        Args:
            patches: List of patch dictionaries
            max_window_size: Maximum window size for padding
            
        Returns:
            Tuple of (padded_values, attention_mask)
        """
        if max_window_size is None:
            max_window_size = max(self.config.window_sizes)
        
        batch_size = len(patches)
        padded_values = np.zeros((batch_size, max_window_size))
        attention_mask = np.zeros((batch_size, max_window_size))
        
        for i, patch in enumerate(patches):
            patch_length = len(patch['values'])
            actual_length = min(patch_length, max_window_size)
            
            padded_values[i, :actual_length] = patch['values'][:actual_length]
            attention_mask[i, :actual_length] = 1.0
        
        return padded_values, attention_mask

class PatchEmbeddingPipeline:
    """
    Complete pipeline for patch embedding.
    
    This class combines the tokenizer and embedder to create embeddings
    for time series patches that can be used in the Time-LlaMA model.
    """
    
    def __init__(self, 
                 patch_config: PatchConfig = None,
                 embedder_config: Dict[str, Any] = None):
        self.patch_config = patch_config or PatchConfig()
        self.embedder_config = embedder_config or {}
        
        self.tokenizer = PatchTokenizer(self.patch_config)
        self.embedder = PatchEmbedder(
            embedding_dim=self.patch_config.embedding_dim,
            max_window_size=max(self.patch_config.window_sizes),
            **self.embedder_config
        )
        
    def fit(self, data: pd.DataFrame):
        """Fit the pipeline on training data"""
        self.tokenizer.fit(data)
        
    def transform(self, data: pd.DataFrame, 
                 target_series: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Transform data into patch embeddings.
        
        Args:
            data: Dataframe with time series
            target_series: List of target series
            
        Returns:
            Dictionary mapping series_id to embedding tensors
        """
        # Create patch sequences
        patch_sequences = self.tokenizer.create_patch_sequences(data, target_series)
        
        embeddings = {}
        
        for series_id, patches in patch_sequences.items():
            if not patches:
                continue
            
            # Pad patches
            padded_values, attention_mask = self.tokenizer.pad_patches(patches)
            
            # Convert to tensors
            patch_tensor = torch.FloatTensor(padded_values).unsqueeze(-1)  # Add input_dim
            mask_tensor = torch.FloatTensor(attention_mask)
            
            # Get embeddings
            with torch.no_grad():
                series_embeddings = self.embedder(patch_tensor)
            
            # Apply attention mask
            series_embeddings = series_embeddings * mask_tensor.unsqueeze(-1)
            
            embeddings[series_id] = {
                'embeddings': series_embeddings,
                'attention_mask': mask_tensor,
                'patches': patches
            }
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension"""
        return self.patch_config.embedding_dim
    
    def get_max_window_size(self) -> int:
        """Get the maximum window size"""
        return max(self.patch_config.window_sizes)


def main():
    """Example usage of the patch embedding system"""
    import pandas as pd
    
    # Load data
    train_df = pd.read_csv('/data/kaggle_projects/commodity_prediction/kaggle_data/train.csv')
    
    # Configure patches
    patch_config = PatchConfig(
        window_sizes=[7, 14, 28],
        strides=[1, 3, 7],
        embedding_dim=128,
        max_patches_per_series=50
    )
    
    # Initialize pipeline
    pipeline = PatchEmbeddingPipeline(
        patch_config=patch_config,
        embedder_config={'architecture': 'hybrid', 'dropout': 0.1}
    )
    
    # Fit on training data
    pipeline.fit(train_df)
    
    # Transform data
    embeddings = pipeline.transform(train_df, target_series=['LME_AL_Close', 'JPX_Gold_Close'])
    
    print(f"Created embeddings for {len(embeddings)} series")
    for series_id, data in embeddings.items():
        print(f"{series_id}: {data['embeddings'].shape}")


if __name__ == "__main__":
    main()
