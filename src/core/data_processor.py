"""
Unified Data Processing Module for Commodity Time Series

This module provides a centralized data processing pipeline that handles:
1. Data loading and validation
2. Time series preprocessing
3. Patch generation and embedding
4. Target pair processing
5. Data splitting and batching
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from .config import SystemConfig, PatchConfig

logger = logging.getLogger(__name__)

class CommodityDataProcessor:
    """
    Unified data processor for commodity time series data.
    
    This class handles all aspects of data processing including loading,
    preprocessing, patch generation, and target preparation.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.scaler = None
        self.target_pairs = None
        self.series_metadata = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all required data files.
        
        Returns:
            Tuple of (train_data, test_data, target_pairs)
        """
        logger.info("Loading data files...")
        
        # Load training data
        train_path = Path(self.config.train_data_path)
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        train_data = pd.read_csv(train_path)
        logger.info(f"Loaded training data: {train_data.shape}")
        
        # Load test data
        test_path = Path(self.config.test_data_path)
        if test_path.exists():
            test_data = pd.read_csv(test_path)
            logger.info(f"Loaded test data: {test_data.shape}")
        else:
            test_data = None
            logger.warning(f"Test data not found: {test_path}")
        
        # Load target pairs
        target_pairs_path = Path(self.config.target_pairs_path)
        if not target_pairs_path.exists():
            raise FileNotFoundError(f"Target pairs not found: {target_pairs_path}")
        
        target_pairs = pd.read_csv(target_pairs_path)
        self.target_pairs = target_pairs
        logger.info(f"Loaded target pairs: {len(target_pairs)} targets")
        
        return train_data, test_data, target_pairs
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the time series data.
        
        Args:
            data: Raw time series data
            
        Returns:
            Preprocessed data
        """
        logger.info("Preprocessing data...")
        
        # Ensure date_id is datetime
        if 'date_id' in data.columns:
            data['date_id'] = pd.to_datetime(data['date_id'])
        
        # Sort by date
        data = data.sort_values('date_id').reset_index(drop=True)
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'date_id':
                # Forward fill then backward fill
                data[col] = data[col].ffill().bfill()
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        cols_to_remove = []
        for col in numeric_columns:
            if col != 'date_id' and data[col].isna().sum() / len(data) > missing_threshold:
                cols_to_remove.append(col)
        
        if cols_to_remove:
            logger.warning(f"Removing columns with >50% missing values: {cols_to_remove}")
            data = data.drop(columns=cols_to_remove)
        
        logger.info(f"Preprocessed data shape: {data.shape}")
        return data
    
    def extract_series_metadata(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Extract metadata for each time series.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary mapping series_id to metadata
        """
        logger.info("Extracting series metadata...")
        
        metadata = {}
        exchange_prefixes = {
            'LME': 'LME_',
            'JPX': 'JPX_', 
            'US': 'US_',
            'FX': 'FX_'
        }
        
        for column in data.columns:
            if column == 'date_id':
                continue
            
            # Parse series identifier
            exchange = 'UNKNOWN'
            instrument = column
            
            for ex, prefix in exchange_prefixes.items():
                if column.startswith(prefix):
                    exchange = ex
                    instrument = column[len(prefix):]
                    
                    # Remove common suffixes
                    suffixes = ['_Close', '_adj_close', '_Volume', '_Open', '_High', '_Low']
                    for suffix in suffixes:
                        if instrument.endswith(suffix):
                            instrument = instrument[:-len(suffix)]
                            break
                    break
            
            # Extract series statistics
            series_data = data[column].dropna()
            
            metadata[column] = {
                'exchange': exchange,
                'instrument': instrument,
                'length': len(series_data),
                'start_date': series_data.index[0] if len(series_data) > 0 else None,
                'end_date': series_data.index[-1] if len(series_data) > 0 else None,
                'mean': series_data.mean(),
                'std': series_data.std(),
                'min': series_data.min(),
                'max': series_data.max(),
                'missing_ratio': data[column].isna().sum() / len(data)
            }
        
        self.series_metadata = metadata
        logger.info(f"Extracted metadata for {len(metadata)} series")
        return metadata
    
    def fit_scaler(self, data: pd.DataFrame) -> 'CommodityDataProcessor':
        """
        Fit the data scaler on training data.
        
        Args:
            data: Training data
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting data scaler...")
        
        # Collect all numeric values for scaling
        all_values = []
        for column in data.columns:
            if column != 'date_id':
                series_data = data[column].dropna()
                all_values.extend(series_data.values)
        
        # Fit scaler
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.scaler.fit(np.array(all_values).reshape(-1, 1))
        
        logger.info("Data scaler fitted")
        return self
    
    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the fitted scaler.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        logger.info("Transforming data...")
        
        transformed_data = data.copy()
        
        for column in data.columns:
            if column != 'date_id':
                # Transform non-null values
                mask = ~data[column].isna()
                if mask.any():
                    values = data.loc[mask, column].values.reshape(-1, 1)
                    transformed_values = self.scaler.transform(values).flatten()
                    transformed_data.loc[mask, column] = transformed_values
        
        logger.info("Data transformation complete")
        return transformed_data
    
    def create_patches(self, data: pd.DataFrame, series_id: str) -> List[Dict[str, Any]]:
        """
        Create patches for a specific time series.
        
        Args:
            data: Time series data
            series_id: Series identifier
            
        Returns:
            List of patch dictionaries
        """
        if series_id not in data.columns:
            logger.warning(f"Series {series_id} not found in data")
            return []
        
        series_data = data[series_id].dropna()
        if len(series_data) < min(self.config.patches.window_sizes):
            logger.warning(f"Insufficient data for {series_id}: {len(series_data)} points")
            return []
        
        patches = []
        
        for window_size in self.config.patches.window_sizes:
            for stride in self.config.patches.strides:
                # Create overlapping windows
                for start_idx in range(0, len(series_data) - window_size + 1, stride):
                    end_idx = start_idx + window_size
                    window_data = series_data.iloc[start_idx:end_idx]
                    
                    # Skip if window is too short
                    if len(window_data) < self.config.patches.min_patch_length:
                        continue
                    
                    # Create patch
                    patch = {
                        'series_id': series_id,
                        'window_size': window_size,
                        'stride': stride,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'start_date': series_data.index[start_idx],
                        'end_date': series_data.index[end_idx - 1],
                        'values': window_data.values,
                        'patch_id': f"{series_id}_{window_size}d_{stride}s_{start_idx}_{end_idx}"
                    }
                    
                    patches.append(patch)
        
        # Limit patches per series
        if len(patches) > self.config.patches.max_patches_per_series:
            # Keep most recent patches
            patches.sort(key=lambda x: x['end_date'])
            patches = patches[-self.config.patches.max_patches_per_series:]
        
        return patches
    
    def create_all_patches(self, data: pd.DataFrame, target_series: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create patches for all time series.
        
        Args:
            data: Time series data
            target_series: List of target series (if None, use all)
            
        Returns:
            Dictionary mapping series_id to list of patches
        """
        if target_series is None:
            target_series = [col for col in data.columns if col != 'date_id']
        
        logger.info(f"Creating patches for {len(target_series)} series...")
        
        all_patches = {}
        total_patches = 0
        
        for series_id in target_series:
            patches = self.create_patches(data, series_id)
            all_patches[series_id] = patches
            total_patches += len(patches)
            
            if len(patches) > 0:
                logger.debug(f"Created {len(patches)} patches for {series_id}")
        
        logger.info(f"Created {total_patches} patches total")
        return all_patches
    
    def prepare_targets(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Prepare target values for all target pairs.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary mapping target to target information
        """
        logger.info("Preparing targets...")
        
        targets = {}
        
        for _, row in self.target_pairs.iterrows():
            target = row['target']
            pair = row['pair']
            lag = row['lag']
            
            # Parse target to get component series
            if ' - ' in target:
                # Difference target
                parts = target.split(' - ')
                series1, series2 = parts[0].strip(), parts[1].strip()
                
                if series1 in data.columns and series2 in data.columns:
                    # Calculate difference
                    target_values = data[series1] - data[series2]
                else:
                    logger.warning(f"Missing series for target {target}")
                    continue
            else:
                # Single series target
                if target in data.columns:
                    target_values = data[target]
                else:
                    logger.warning(f"Missing series for target {target}")
                    continue
            
            targets[target] = {
                'pair': pair,
                'lag': lag,
                'values': target_values,
                'type': 'difference' if ' - ' in target else 'single'
            }
        
        logger.info(f"Prepared {len(targets)} targets")
        return targets
    
    def split_data(self, data: pd.DataFrame, val_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and validation sets.
        
        Args:
            data: Time series data
            val_split: Validation split ratio
            
        Returns:
            Tuple of (train_data, val_data)
        """
        logger.info(f"Splitting data with {val_split} validation ratio...")
        
        # Sort by date to ensure temporal order
        data = data.sort_values('date_id').reset_index(drop=True)
        
        # Split by date to maintain temporal order
        split_idx = int(len(data) * (1 - val_split))
        
        train_data = data.iloc[:split_idx].copy()
        val_data = data.iloc[split_idx:].copy()
        
        logger.info(f"Split data: train={len(train_data)}, val={len(val_data)}")
        return train_data, val_data

class CommodityDataset(Dataset):
    """
    PyTorch Dataset for commodity time series forecasting.
    
    This dataset handles the creation of training samples with patches,
    targets, and metadata for the hybrid forecasting system.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 target_pairs: pd.DataFrame,
                 data_processor: CommodityDataProcessor,
                 config: SystemConfig,
                 mode: str = 'train'):
        self.data = data
        self.target_pairs = target_pairs
        self.data_processor = data_processor
        self.config = config
        self.mode = mode
        
        # Create training samples
        self.samples = self._create_samples()
        
    def _create_samples(self) -> List[Dict[str, Any]]:
        """Create training samples from the data"""
        logger.info(f"Creating {self.mode} samples...")
        
        samples = []
        dates = sorted(self.data['date_id'].unique())
        
        # Create samples for each target pair
        for _, target_row in self.target_pairs.iterrows():
            target = target_row['target']
            pair = target_row['pair']
            lag = target_row['lag']
            
            # Parse target series
            target_series = self._parse_target(target)
            
            # Create samples for different forecast dates
            for i in range(self.config.training.min_lookback, len(dates) - lag):
                forecast_date = dates[i + lag]
                lookback_date = dates[i]
                
                # Get lookback data
                lookback_data = self.data[self.data['date_id'] <= lookback_date]
                
                if len(lookback_data) < self.config.training.min_lookback_days:
                    continue
                
                # Get target value
                target_data = self.data[self.data['date_id'] == forecast_date]
                if target_data.empty:
                    continue
                
                target_value = target_data[target].iloc[0] if target in target_data.columns else None
                if target_value is None or pd.isna(target_value):
                    continue
                
                sample = {
                    'target': target,
                    'pair': pair,
                    'lag': lag,
                    'forecast_date': forecast_date,
                    'lookback_date': lookback_date,
                    'target_value': target_value,
                    'lookback_data': lookback_data
                }
                
                samples.append(sample)
        
        logger.info(f"Created {len(samples)} {self.mode} samples")
        return samples
    
    def _parse_target(self, target: str) -> str:
        """Parse target string to get primary series"""
        if ' - ' in target:
            parts = target.split(' - ')
            return parts[0].strip()
        else:
            return target
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample"""
        sample = self.samples[idx]
        
        # Get patch embeddings
        patch_embeddings = self.data_processor.create_all_patches(
            sample['lookback_data'],
            target_series=[sample['target']]
        )
        
        if sample['target'] not in patch_embeddings or not patch_embeddings[sample['target']]:
            # Create dummy embeddings if no patches available
            dummy_patch = {
                'values': np.zeros(min(self.config.patches.window_sizes)),
                'patch_id': f"dummy_{sample['target']}"
            }
            patch_embeddings = {sample['target']: [dummy_patch]}
        
        return {
            'patches': patch_embeddings[sample['target']],
            'target_value': sample['target_value'],
            'target': sample['target'],
            'lag': sample['lag'],
            'forecast_date': sample['forecast_date'],
            'lookback_data': sample['lookback_data']
        }

def create_data_loaders(train_data: pd.DataFrame,
                       val_data: pd.DataFrame,
                       target_pairs: pd.DataFrame,
                       data_processor: CommodityDataProcessor,
                       config: SystemConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        train_data: Training data
        val_data: Validation data
        target_pairs: Target pairs configuration
        data_processor: Data processor instance
        config: System configuration
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = CommodityDataset(train_data, target_pairs, data_processor, config, mode='train')
    val_dataset = CommodityDataset(val_data, target_pairs, data_processor, config, mode='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for batching samples with proper handling of variable-length sequences.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched data with proper padding and masking
    """
    batch_size = len(batch)
    
    # Extract basic information
    target_values = torch.tensor([sample['target_value'] for sample in batch], dtype=torch.float32)
    targets = [sample['target'] for sample in batch]
    lags = torch.tensor([sample['lag'] for sample in batch], dtype=torch.long)
    forecast_dates = [sample['forecast_date'] for sample in batch]
    
    # Handle patches - convert to tensors with padding
    all_patches = []
    patch_masks = []
    max_patches = 0
    
    for sample in batch:
        patches = sample['patches']
        if not patches:
            # Create dummy patch if no patches available
            patches = [{'values': np.zeros(7), 'patch_id': 'dummy'}]
        
        # Convert patch values to tensors
        patch_tensors = []
        for patch in patches:
            if isinstance(patch['values'], np.ndarray):
                patch_tensor = torch.from_numpy(patch['values']).float()
            else:
                patch_tensor = torch.tensor(patch['values'], dtype=torch.float32)
            patch_tensors.append(patch_tensor)
        
        all_patches.append(patch_tensors)
        max_patches = max(max_patches, len(patch_tensors))
    
    # Pad patches to same length
    padded_patches = []
    for patch_list in all_patches:
        # Pad with zeros if needed
        while len(patch_list) < max_patches:
            # Create dummy patch with same shape as first patch
            if patch_list:
                dummy_patch = torch.zeros_like(patch_list[0])
            else:
                dummy_patch = torch.zeros(7)  # Default patch size
            patch_list.append(dummy_patch)
        
        # Stack patches for this sample
        sample_patches = torch.stack(patch_list)
        padded_patches.append(sample_patches)
    
    # Stack all samples
    if padded_patches:
        batched_patches = torch.stack(padded_patches)
    else:
        # Fallback if no patches
        batched_patches = torch.zeros(batch_size, 1, 7)
    
    # Create attention mask (1 for real patches, 0 for padded)
    attention_masks = []
    for i, patch_list in enumerate(all_patches):
        mask = torch.ones(len(patch_list))
        if len(patch_list) < max_patches:
            # Pad mask with zeros
            mask = torch.cat([mask, torch.zeros(max_patches - len(patch_list))])
        attention_masks.append(mask)
    
    attention_mask = torch.stack(attention_masks)
    
    batched_data = {
        'patches': batched_patches,
        'attention_mask': attention_mask,
        'target_values': target_values,
        'targets': targets,
        'lags': lags,
        'forecast_dates': forecast_dates
    }
    
    return batched_data
