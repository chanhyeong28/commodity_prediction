"""
Time-LlaMA DataLoader

This dataloader implements the Time-LlaMA approach with:
- Direct input-to-output mapping (no label_len overlap)
- Channel-as-token embedding (NO time features needed)
- Cross-attention alignment instead of sequential processing
- Simplified data loading focused on time series values only
- Support for both standard time series and Mitsui commodity data formats

Key difference from Time-LLM: Time-LlaMA does NOT use time features (x_mark)
because it uses channel-as-token embedding and cross-attention with prompts
for temporal context instead of explicit time feature engineering.
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings

warnings.filterwarnings('ignore')

class Dataset_Mitsui_Commodity(Dataset):
    """
    Mitsui commodity prediction dataset for TimeLlaMA.
    
    Key differences from Time-LLM:
    - Direct input-to-output mapping without label_len overlap
    - NO time features (x_mark) - uses channel-as-token embedding instead
    - Channel-as-token approach: Each time series channel becomes one token
    - Process ALL channels together in one forward pass (not feature-wise)
    - Simplified data loading focused on time series values only
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='train.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, few_shot_ratio=1.0):
        if size == None:
            # Time-LlaMA default sizes for commodity data
            self.seq_len = 96   # 4 days of hourly data
            self.pred_len = 24  # 1 day prediction
        else:
            self.seq_len = size[0]
            self.pred_len = size[2]  # Only use seq_len and pred_len
        
        # Time-LlaMA: No label_len needed (direct input-to-output mapping)
        self.label_len = 0  # Explicitly set to 0 for Time-LlaMA
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        # Clamp ratios to safe ranges
        self.percent = max(1, min(int(percent), 100))
        self.few_shot_ratio = float(max(0.0, min(float(few_shot_ratio), 1.0)))

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]  # Number of channels
        # Time-LlaMA: No need for tot_len since we process all channels together

    def __read_data__(self):
        """
        Read and preprocess data for Time-LlaMA.
        Handles both standard time series format and Mitsui commodity format.
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Handle different data formats
        if 'date_id' in df_raw.columns:
            # Mitsui commodity data format
            # Use all columns except the identifier as features
            value_cols = [c for c in df_raw.columns if c != 'date_id']
            df_raw = df_raw[['date_id'] + value_cols]
        else:
            # Standard time series format
            '''
            df_raw.columns: ['date', ...(other features), target feature]
            '''
            cols = list(df_raw.columns)
            cols.remove(self.target)
            if 'date' in cols:
                cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            # Apply percentage and few-shot ratio for training data
            effective_percent = self.percent * self.few_shot_ratio
            border2 = (border2 - self.seq_len) * effective_percent // 100 + self.seq_len

        # Select data columns (exclude date columns if present)
        if 'date' in df_raw.columns:
            data_cols = [c for c in df_raw.columns if c not in ('date', 'date_id')]
        else:
            data_cols = [c for c in df_raw.columns if c != 'date_id']

        if self.features == 'S' and self.target in df_raw.columns:
            df_data = df_raw[[self.target]]
        else:
            # Multivariate by default
            df_data = df_raw[data_cols]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values).astype(np.float32)
        else:
            data = df_data.values.astype(np.float32)

        # Time-LlaMA uses channel-as-token embedding, not time features
        # Skip time feature computation for efficiency
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = None  # Not used in Time-LlaMA
        
        # Validate data for Time-LlaMA requirements
        self._validate_data()
    
    def _validate_data(self):
        """
        Validate data for Time-LlaMA requirements.
        """
        # Check minimum data length
        min_required_length = self.seq_len + self.pred_len
        if len(self.data_x) < min_required_length:
            raise ValueError(f"Data too short: {len(self.data_x)} < {min_required_length} (seq_len + pred_len)")
        
        # Check for NaN values
        if np.isnan(self.data_x).any():
            warnings.warn("Data contains NaN values, filling with 0")
            self.data_x = np.nan_to_num(self.data_x, nan=0.0)
            self.data_y = np.nan_to_num(self.data_y, nan=0.0)
        
        # Check for infinite values
        if np.isinf(self.data_x).any():
            warnings.warn("Data contains infinite values, clipping to finite range")
            self.data_x = np.clip(self.data_x, -1e10, 1e10)
            self.data_y = np.clip(self.data_y, -1e10, 1e10)
        
        # Validate channel count
        if self.data_x.shape[1] == 0:
            raise ValueError("No features found in data")
        
        # Optional log for debugging
        # print(f"Time-LlaMA Data: {len(self.data_x)} samples, {self.data_x.shape[1]} channels")

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        
        Time-LlaMA Channel-as-Token Approach:
        - Each time series channel becomes one token
        - Process all channels together in one forward pass
        - No feature-wise processing like Time-LLM
        
        Returns:
            seq_x: Input sequence [seq_len, num_channels] - ALL channels as tokens
            seq_y: Target sequence [pred_len, num_channels] - ALL channels as tokens
            None: Time features not used in Time-LlaMA (channel-as-token approach)
            None: Time features not used in Time-LlaMA (channel-as-token approach)
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end  # No overlap - direct prediction (Time-LlaMA approach)
        r_end = r_begin + self.pred_len
        
        # Channel-as-token: Process ALL channels together
        seq_x = self.data_x[s_begin:s_end, :]  # [seq_len, num_channels]
        seq_y = self.data_y[r_begin:r_end, :]  # [pred_len, num_channels]
        
        # Time-LlaMA uses channel-as-token embedding, not time features
        # Return None for time features to maintain compatibility with existing code
        return seq_x, seq_y, None, None

    def __len__(self):
        # Time-LlaMA: Channel-as-token approach processes all channels together
        # No need to multiply by enc_in (number of channels)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def get_dataset_stats(self):
        """
        Get dataset statistics for Time-LlaMA analysis.
        """
        stats = {
            'num_samples': len(self.data_x),
            'num_channels': self.data_x.shape[1],
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'label_len': self.label_len,  # Should be 0 for Time-LlaMA
            'data_shape': self.data_x.shape,
            'min_value': float(np.min(self.data_x)),
            'max_value': float(np.max(self.data_x)),
            'mean_value': float(np.mean(self.data_x)),
            'std_value': float(np.std(self.data_x)),
            'few_shot_ratio': self.few_shot_ratio
        }
        return stats

