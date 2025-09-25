import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
import warnings


class TimeSeriesDataset(Dataset):
    """
    Time series dataset for TimeLlaMA training and evaluation.
    
    Handles the Mitsui commodity prediction dataset format.
    """
    
    def __init__(
        self,
        data_path: str,
        flag: str = 'train',
        size: Tuple[int, int, int] = (96, 48, 96),
        features: str = 'M',
        target: str = 'OT',
        scale: bool = True,
        timeenc: int = 0,
        freq: str = 'h',
        seasonal_patterns: Optional[str] = None
    ):
        """
        Initialize TimeSeriesDataset.
        
        Args:
            data_path: Path to the data file
            flag: 'train', 'test', or 'val'
            size: [seq_len, label_len, pred_len]
            features: 'M' (multivariate), 'S' (univariate), 'MS' (multivariate with single target)
            target: Target column name
            scale: Whether to apply scaling
            timeenc: Time encoding type (0: no time features, 1: time features)
            freq: Frequency of the time series
            seasonal_patterns: Seasonal patterns for the dataset
        """
        assert flag in ['train', 'test', 'val']
        
        self.flag = flag
        self.size = size
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.seasonal_patterns = seasonal_patterns
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # Load and preprocess data
        self.data_x, self.data_y, self.data_stamp = self._load_data(data_path)
        
        # Apply scaling if needed
        if self.scale:
            self._apply_scaling()
    
    def _load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load data from CSV file."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load CSV data
        df_raw = pd.read_csv(data_path)
        
        # Handle different data formats
        if 'date' in df_raw.columns:
            # Standard time series format
            df_raw['date'] = pd.to_datetime(df_raw['date'])
            df_raw = df_raw.sort_values('date').reset_index(drop=True)
            cols = list(df_raw.columns)
            cols.remove('date')
            df_data = df_raw[cols]
        else:
            # Assume first column is time index or no time column
            df_data = df_raw
        
        # Handle target column
        if self.target in df_data.columns:
            target_idx = df_data.columns.get_loc(self.target)
        else:
            # Use last column as target if target not found
            target_idx = -1
            warnings.warn(f"Target column '{self.target}' not found. Using last column as target.")
        
        # Select features based on feature type
        if self.features == 'M' or self.features == 'MS':
            # Multivariate: use all columns
            data = df_data.values
        elif self.features == 'S':
            # Univariate: use only target column
            data = df_data.iloc[:, [target_idx]].values
        else:
            raise ValueError(f"Unknown features type: {self.features}")
        
        # Split data based on flag
        if self.flag == 'train':
            data = data[:int(len(data) * 0.7)]
        elif self.flag == 'val':
            data = data[int(len(data) * 0.7):int(len(data) * 0.9)]
        else:  # test
            data = data[int(len(data) * 0.9):]
        
        # Create time stamps
        if self.timeenc == 1:
            # Create time features
            time_stamps = self._create_time_features(len(data))
        else:
            # No time features
            time_stamps = np.zeros((len(data), 1))
        
        # Prepare input and target data
        data_x = data
        data_y = data  # For forecasting, target is the same as input
        
        return data_x, data_y, time_stamps
    
    def _create_time_features(self, length: int) -> np.ndarray:
        """Create time features."""
        # Simple time features: hour, day, month, year
        # This is a simplified version - in practice, you'd use proper datetime features
        time_features = np.zeros((length, 4))
        
        for i in range(length):
            # Simulate time features (in practice, use actual datetime)
            time_features[i, 0] = i % 24  # hour
            time_features[i, 1] = (i // 24) % 7  # day of week
            time_features[i, 2] = (i // (24 * 7)) % 12  # month
            time_features[i, 3] = i // (24 * 7 * 12)  # year
        
        return time_features
    
    def _apply_scaling(self):
        """Apply scaling to the data."""
        if self.flag == 'train':
            # Calculate scaling parameters on training data
            self.scaler_mean = np.mean(self.data_x, axis=0)
            self.scaler_std = np.std(self.data_x, axis=0)
            # Avoid division by zero
            self.scaler_std[self.scaler_std == 0] = 1.0
        else:
            # Use scaling parameters from training (should be set externally)
            if not hasattr(self, 'scaler_mean') or not hasattr(self, 'scaler_std'):
                warnings.warn("Scaling parameters not found. Data will not be scaled.")
                return
        
        # Apply scaling
        self.data_x = (self.data_x - self.scaler_mean) / self.scaler_std
        self.data_y = (self.data_y - self.scaler_mean) / self.scaler_std
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        if hasattr(self, 'scaler_mean') and hasattr(self, 'scaler_std'):
            return data * self.scaler_std + self.scaler_mean
        return data
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            batch_x: Input sequence [seq_len, features]
            batch_y: Target sequence [pred_len, features]
            batch_x_mark: Input time features [seq_len, time_features]
            batch_y_mark: Target time features [pred_len, time_features]
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # Input sequence
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        # Time features
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return (
            torch.FloatTensor(seq_x),
            torch.FloatTensor(seq_y),
            torch.FloatTensor(seq_x_mark),
            torch.FloatTensor(seq_y_mark)
        )


def data_provider(args, flag: str) -> Tuple[TimeSeriesDataset, DataLoader]:
    """
    Data provider function for TimeLlaMA.
    
    Args:
        args: Arguments containing data configuration
        flag: 'train', 'val', or 'test'
        
    Returns:
        Tuple of (dataset, dataloader)
    """
    # Determine data path
    if hasattr(args, 'data_path'):
        data_path = args.data_path
    else:
        # Default path for Mitsui dataset
        data_path = '/Users/minkeychang/commodity_prediction/kaggle/train.csv'
    
    # Create dataset
    dataset = TimeSeriesDataset(
        data_path=data_path,
        flag=flag,
        size=(args.seq_len, args.label_len, args.pred_len),
        features=args.features,
        target=args.target,
        scale=args.scale,
        timeenc=args.timeenc,
        freq=args.freq,
        seasonal_patterns=getattr(args, 'seasonal_patterns', None)
    )
    
    # Create dataloader
    shuffle = (flag == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0,
        drop_last=True
    )
    
        return dataset, dataloader


class MitsuiDataset(Dataset):
    """
    Mitsui commodity prediction dataset for TimeLlaMA.
    
    Handles the specific format of the Mitsui competition dataset.
    """
    
    def __init__(
        self,
        data_path: str,
        labels_path: str,
        flag: str = 'train',
        size: Tuple[int, int, int] = (96, 48, 96),
        features: str = 'M',
        target: str = 'OT',
        scale: bool = True,
        timeenc: int = 0,
        freq: str = 'h',
        max_targets: int = 50  # Limit number of targets for memory efficiency
    ):
        """
        Initialize MitsuiDataset.
        
        Args:
            data_path: Path to train.csv
            labels_path: Path to train_labels.csv
            flag: 'train', 'test', or 'val'
            size: [seq_len, label_len, pred_len]
            features: 'M' (multivariate), 'S' (univariate), 'MS' (multivariate with single target)
            target: Target column name (not used for Mitsui)
            scale: Whether to apply scaling
            timeenc: Time encoding type (0: no time features, 1: time features)
            freq: Frequency of the time series
            max_targets: Maximum number of targets to use (for memory efficiency)
        """
        assert flag in ['train', 'test', 'val']
        
        self.flag = flag
        self.size = size
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.max_targets = max_targets
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # Load and preprocess data
        self.data_x, self.data_y, self.data_stamp = self._load_mitsui_data(data_path, labels_path)
        
        # Apply scaling if needed
        if self.scale:
            self._apply_scaling()
    
    def _load_mitsui_data(self, data_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load Mitsui dataset."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        # Load main data
        df_data = pd.read_csv(data_path)
        
        # Remove date_id column (we'll use it for temporal ordering)
        if 'date_id' in df_data.columns:
            df_data = df_data.drop('date_id', axis=1)
        
        # Load labels
        df_labels = pd.read_csv(labels_path)
        
        # Remove date_id from labels
        if 'date_id' in df_labels.columns:
            df_labels = df_labels.drop('date_id', axis=1)
        
        # Limit number of targets for memory efficiency
        if self.max_targets > 0 and len(df_labels.columns) > self.max_targets:
            df_labels = df_labels.iloc[:, :self.max_targets]
            print(f"Limited to {self.max_targets} targets for memory efficiency")
        
        # Handle missing values in data
        df_data = df_data.fillna(0.0)  # Fill missing values with 0
        
        # Handle missing values in labels
        df_labels = df_labels.fillna(0.0)  # Fill missing values with 0
        
        # Convert to numpy arrays
        data_x = df_data.values.astype(np.float32)
        data_y = df_labels.values.astype(np.float32)
        
        # Ensure same length
        min_length = min(len(data_x), len(data_y))
        data_x = data_x[:min_length]
        data_y = data_y[:min_length]
        
        # Split data based on flag (temporal split)
        total_length = len(data_x)
        if self.flag == 'train':
            # Use first 70% for training
            end_idx = int(total_length * 0.7)
            data_x = data_x[:end_idx]
            data_y = data_y[:end_idx]
        elif self.flag == 'val':
            # Use 70-90% for validation
            start_idx = int(total_length * 0.7)
            end_idx = int(total_length * 0.9)
            data_x = data_x[start_idx:end_idx]
            data_y = data_y[start_idx:end_idx]
        else:  # test
            # Use last 10% for testing
            start_idx = int(total_length * 0.9)
            data_x = data_x[start_idx:]
            data_y = data_y[start_idx:]
        
        # Create time stamps
        if self.timeenc == 1:
            # Create time features
            time_stamps = self._create_time_features(len(data_x))
        else:
            # No time features
            time_stamps = np.zeros((len(data_x), 1))
        
        return data_x, data_y, time_stamps
    
    def _create_time_features(self, length: int) -> np.ndarray:
        """Create time features for Mitsui dataset."""
        # Simple time features: hour, day, month, year
        time_features = np.zeros((length, 4))
        
        for i in range(length):
            # Simulate time features (in practice, use actual datetime)
            time_features[i, 0] = i % 24  # hour
            time_features[i, 1] = (i // 24) % 7  # day of week
            time_features[i, 2] = (i // (24 * 7)) % 12  # month
            time_features[i, 3] = i // (24 * 7 * 12)  # year
        
        return time_features
    
    def _apply_scaling(self):
        """Apply scaling to the data."""
        if self.flag == 'train':
            # Calculate scaling parameters on training data
            self.scaler_mean_x = np.mean(self.data_x, axis=0)
            self.scaler_std_x = np.std(self.data_x, axis=0)
            self.scaler_mean_y = np.mean(self.data_y, axis=0)
            self.scaler_std_y = np.std(self.data_y, axis=0)
            
            # Avoid division by zero
            self.scaler_std_x[self.scaler_std_x == 0] = 1.0
            self.scaler_std_y[self.scaler_std_y == 0] = 1.0
        else:
            # Use scaling parameters from training (should be set externally)
            if not hasattr(self, 'scaler_mean_x') or not hasattr(self, 'scaler_std_x'):
                warnings.warn("Scaling parameters not found. Data will not be scaled.")
                return
        
        # Apply scaling
        self.data_x = (self.data_x - self.scaler_mean_x) / self.scaler_std_x
        self.data_y = (self.data_y - self.scaler_mean_y) / self.scaler_std_y
    
    def inverse_transform(self, data: np.ndarray, is_target: bool = False) -> np.ndarray:
        """Inverse transform scaled data."""
        if is_target:
            if hasattr(self, 'scaler_mean_y') and hasattr(self, 'scaler_std_y'):
                return data * self.scaler_std_y + self.scaler_mean_y
        else:
            if hasattr(self, 'scaler_mean_x') and hasattr(self, 'scaler_std_x'):
                return data * self.scaler_std_x + self.scaler_mean_x
        return data
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            batch_x: Input sequence [seq_len, features]
            batch_y: Target sequence [pred_len, targets]
            batch_x_mark: Input time features [seq_len, time_features]
            batch_y_mark: Target time features [pred_len, time_features]
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # Input sequence
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        # Time features
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return (
            torch.FloatTensor(seq_x),
            torch.FloatTensor(seq_y),
            torch.FloatTensor(seq_x_mark),
            torch.FloatTensor(seq_y_mark)
        )


def mitsui_data_provider(args, flag: str) -> Tuple[MitsuiDataset, DataLoader]:
    """
    Mitsui data provider function for TimeLlaMA.
    
    Args:
        args: Arguments containing data configuration
        flag: 'train', 'val', or 'test'
        
    Returns:
        Tuple of (dataset, dataloader)
    """
    # Determine data paths
    if hasattr(args, 'data_path'):
        data_path = args.data_path
    else:
        data_path = '/Users/minkeychang/commodity_prediction/kaggle/train.csv'
    
    if hasattr(args, 'labels_path'):
        labels_path = args.labels_path
    else:
        labels_path = '/Users/minkeychang/commodity_prediction/kaggle/train_labels.csv'
    
    # Create dataset
    dataset = MitsuiDataset(
        data_path=data_path,
        labels_path=labels_path,
        flag=flag,
        size=(args.seq_len, args.label_len, args.pred_len),
        features=args.features,
        target=args.target,
        scale=args.scale,
        timeenc=args.timeenc,
        freq=args.freq,
        max_targets=getattr(args, 'max_targets', 50)
    )
    
    # Create dataloader
    shuffle = (flag == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0,
        drop_last=True
    )
    
    return dataset, dataloader


# For backward compatibility
def data_factory(args, flag: str) -> Tuple[TimeSeriesDataset, DataLoader]:
    """Alias for data_provider for backward compatibility."""
    return data_provider(args, flag)
