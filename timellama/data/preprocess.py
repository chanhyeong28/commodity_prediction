import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings


class TimeSeriesPreprocessor:
    """
    Time series preprocessing utilities for TimeLlaMA.
    
    Handles normalization, feature engineering, and data augmentation.
    """
    
    def __init__(
        self,
        scaler_type: str = 'standard',
        handle_missing: str = 'interpolate',
        add_time_features: bool = True,
        add_lag_features: bool = False,
        max_lags: int = 5
    ):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            handle_missing: How to handle missing values ('interpolate', 'drop', 'fill_zero')
            add_time_features: Whether to add time-based features
            add_lag_features: Whether to add lag features
            max_lags: Maximum number of lag features to add
        """
        self.scaler_type = scaler_type
        self.handle_missing = handle_missing
        self.add_time_features = add_time_features
        self.add_lag_features = add_lag_features
        self.max_lags = max_lags
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.is_fitted = False
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data."""
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Add features
        if self.add_time_features:
            data = self._add_time_features(data)
        
        if self.add_lag_features:
            data = self._add_lag_features(data)
        
        # Fit and transform
        transformed_data = self.scaler.fit_transform(data)
        self.is_fitted = True
        
        return transformed_data
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Add features
        if self.add_time_features:
            data = self._add_time_features(data)
        
        if self.add_lag_features:
            data = self._add_lag_features(data)
        
        # Transform
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
        
        return self.scaler.inverse_transform(data)
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Handle missing values in the data."""
        if self.handle_missing == 'interpolate':
            # Linear interpolation
            df = pd.DataFrame(data)
            df = df.interpolate(method='linear', limit_direction='both')
            return df.values
        elif self.handle_missing == 'drop':
            # Drop rows with missing values
            df = pd.DataFrame(data)
            df = df.dropna()
            return df.values
        elif self.handle_missing == 'fill_zero':
            # Fill with zeros
            return np.nan_to_num(data, nan=0.0)
        else:
            return data
    
    def _add_time_features(self, data: np.ndarray) -> np.ndarray:
        """Add time-based features."""
        # This is a simplified version - in practice, you'd use actual datetime
        length = len(data)
        time_features = np.zeros((length, 4))
        
        for i in range(length):
            time_features[i, 0] = i % 24  # hour
            time_features[i, 1] = (i // 24) % 7  # day of week
            time_features[i, 2] = (i // (24 * 7)) % 12  # month
            time_features[i, 3] = i // (24 * 7 * 12)  # year
        
        # Normalize time features
        time_features = time_features / np.max(time_features, axis=0, keepdims=True)
        
        # Concatenate with original data
        return np.concatenate([data, time_features], axis=1)
    
    def _add_lag_features(self, data: np.ndarray) -> np.ndarray:
        """Add lag features."""
        if self.max_lags <= 0:
            return data
        
        lag_features = []
        for lag in range(1, self.max_lags + 1):
            lag_data = np.roll(data, lag, axis=0)
            lag_data[:lag] = 0  # Set first lag values to 0
            lag_features.append(lag_data)
        
        if lag_features:
            return np.concatenate([data] + lag_features, axis=1)
        return data


class WindowGenerator:
    """
    Generate sliding windows for time series data.
    """
    
    def __init__(
        self,
        input_width: int,
        label_width: int,
        shift: int,
        stride: int = 1
    ):
        """
        Initialize window generator.
        
        Args:
            input_width: Width of input window
            label_width: Width of label window
            shift: Shift between input and label windows
            stride: Stride for window generation
        """
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.stride = stride
    
    def generate_windows(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sliding windows from time series data.
        
        Args:
            data: Input time series data [length, features]
            
        Returns:
            Tuple of (input_windows, label_windows)
        """
        length = len(data)
        
        # Calculate number of windows
        num_windows = (length - self.input_width - self.shift - self.label_width) // self.stride + 1
        
        if num_windows <= 0:
            raise ValueError("Not enough data to generate windows")
        
        # Initialize arrays
        input_windows = np.zeros((num_windows, self.input_width, data.shape[1]))
        label_windows = np.zeros((num_windows, self.label_width, data.shape[1]))
        
        # Generate windows
        for i in range(num_windows):
            start_idx = i * self.stride
            input_windows[i] = data[start_idx:start_idx + self.input_width]
            label_start = start_idx + self.input_width + self.shift
            label_windows[i] = data[label_start:label_start + self.label_width]
        
        return input_windows, label_windows


def create_time_features(
    length: int,
    freq: str = 'h',
    include_cyclical: bool = True
) -> np.ndarray:
    """
    Create time features for time series data.
    
    Args:
        length: Length of the time series
        freq: Frequency ('h' for hourly, 'd' for daily, etc.)
        include_cyclical: Whether to include cyclical encoding
        
    Returns:
        Time features array
    """
    # Create time index
    if freq == 'h':
        periods_per_day = 24
        periods_per_week = 24 * 7
        periods_per_month = 24 * 30
        periods_per_year = 24 * 365
    elif freq == 'd':
        periods_per_day = 1
        periods_per_week = 7
        periods_per_month = 30
        periods_per_year = 365
    else:
        # Default to hourly
        periods_per_day = 24
        periods_per_week = 24 * 7
        periods_per_month = 24 * 30
        periods_per_year = 24 * 365
    
    # Create basic time features
    time_features = np.zeros((length, 4))
    
    for i in range(length):
        time_features[i, 0] = i % periods_per_day  # hour/day
        time_features[i, 1] = (i // periods_per_day) % 7  # day of week
        time_features[i, 2] = (i // periods_per_week) % 4  # week of month
        time_features[i, 3] = (i // periods_per_month) % 12  # month
    
    if include_cyclical:
        # Add cyclical encoding
        cyclical_features = np.zeros((length, 8))
        
        for i in range(length):
            # Hour/day cyclical
            hour = i % periods_per_day
            cyclical_features[i, 0] = np.sin(2 * np.pi * hour / periods_per_day)
            cyclical_features[i, 1] = np.cos(2 * np.pi * hour / periods_per_day)
            
            # Day of week cyclical
            day = (i // periods_per_day) % 7
            cyclical_features[i, 2] = np.sin(2 * np.pi * day / 7)
            cyclical_features[i, 3] = np.cos(2 * np.pi * day / 7)
            
            # Week of month cyclical
            week = (i // periods_per_week) % 4
            cyclical_features[i, 4] = np.sin(2 * np.pi * week / 4)
            cyclical_features[i, 5] = np.cos(2 * np.pi * week / 4)
            
            # Month cyclical
            month = (i // periods_per_month) % 12
            cyclical_features[i, 6] = np.sin(2 * np.pi * month / 12)
            cyclical_features[i, 7] = np.cos(2 * np.pi * month / 12)
        
        # Combine basic and cyclical features
        time_features = np.concatenate([time_features, cyclical_features], axis=1)
    
    return time_features


def normalize_data(
    data: np.ndarray,
    method: str = 'standard',
    fit_data: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Any]:
    """
    Normalize time series data.
    
    Args:
        data: Data to normalize
        method: Normalization method ('standard', 'minmax', 'robust')
        fit_data: Data to fit the scaler (if None, use data)
        
    Returns:
        Tuple of (normalized_data, scaler)
    """
    if fit_data is None:
        fit_data = data
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Fit on fit_data and transform data
    scaler.fit(fit_data)
    normalized_data = scaler.transform(data)
    
    return normalized_data, scaler


def create_lag_features(
    data: np.ndarray,
    lags: List[int],
    fill_method: str = 'zero'
) -> np.ndarray:
    """
    Create lag features for time series data.
    
    Args:
        data: Input time series data
        lags: List of lag values to create
        fill_method: How to fill missing values ('zero', 'mean', 'last')
        
    Returns:
        Data with lag features
    """
    lag_features = []
    
    for lag in lags:
        lag_data = np.roll(data, lag, axis=0)
        
        # Handle missing values at the beginning
        if fill_method == 'zero':
            lag_data[:lag] = 0
        elif fill_method == 'mean':
            lag_data[:lag] = np.mean(data, axis=0)
        elif fill_method == 'last':
            lag_data[:lag] = data[0]
        
        lag_features.append(lag_data)
    
    if lag_features:
        return np.concatenate([data] + lag_features, axis=1)
    return data


def split_time_series(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time series data into train/validation/test sets.
    
    Args:
        data: Input time series data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    length = len(data)
    train_end = int(length * train_ratio)
    val_end = int(length * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data
