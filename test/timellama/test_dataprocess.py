#!/usr/bin/env python3
"""
Test script for data processing and loading functionality.

This script tests the data processing components including:
- TimeSeriesDataset
- MitsuiDataset
- Data preprocessing
- Data loading and batching
- Time feature generation
- Data scaling and normalization
- Train/validation/test splitting
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any

# Add the timellama module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../timellama'))

try:
    from data.dataloader import TimeSeriesDataset, MitsuiDataset, data_provider, mitsui_data_provider
    from data.preprocess import TimeSeriesPreprocessor, WindowGenerator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


def create_mock_data():
    """Create mock time series data for testing."""
    print("Creating mock data...")
    
    # Create mock time series data
    np.random.seed(42)
    n_samples = 1000
    n_features = 21
    n_targets = 50
    
    # Generate time series data
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Add some temporal structure
    for i in range(n_features):
        data[:, i] += np.sin(np.linspace(0, 4*np.pi, n_samples)) * (i + 1) * 0.1
    
    # Generate target data
    targets = np.random.randn(n_samples, n_targets).astype(np.float32)
    
    # Add some correlation with input data
    for i in range(min(n_features, n_targets)):
        targets[:, i] += data[:, i] * 0.5 + np.random.randn(n_samples) * 0.1
    
    return data, targets


def test_timeseries_dataset():
    """Test TimeSeriesDataset functionality."""
    print("Testing TimeSeriesDataset...")
    
    try:
        # Create mock data
        data, targets = create_mock_data()
        
        # Test dataset initialization
        dataset = TimeSeriesDataset(
            data=data,
            targets=targets,
            flag='train',
            size=(96, 48, 96),  # seq_len, label_len, pred_len
            features='M',
            target='OT',
            scale=True,
            timeenc=1,
            freq='h'
        )
        
        print(f"✅ Dataset initialized: {len(dataset)} samples")
        
        # Test data loading
        sample = dataset[0]
        batch_x, batch_y, batch_x_mark, batch_y_mark = sample
        
        expected_x_shape = (96, 21)  # seq_len, n_features
        expected_y_shape = (144, 50)  # label_len + pred_len, n_targets
        
        if batch_x.shape == expected_x_shape and batch_y.shape == expected_y_shape:
            print(f"✅ Sample shapes correct: x={batch_x.shape}, y={batch_y.shape}")
        else:
            print(f"❌ Sample shapes incorrect: x={batch_x.shape}, y={batch_y.shape}")
            return False
        
        # Test scaling
        if hasattr(dataset, 'scaler_mean_x') and hasattr(dataset, 'scaler_std_x'):
            print("✅ Scaling parameters available")
        else:
            print("❌ Scaling parameters not available")
            return False
        
        # Test inverse transform
        original_x = dataset.inverse_transform(batch_x.numpy(), is_target=False)
        original_y = dataset.inverse_transform(batch_y.numpy(), is_target=True)
        
        if original_x.shape == batch_x.shape and original_y.shape == batch_y.shape:
            print("✅ Inverse transform works")
        else:
            print("❌ Inverse transform error")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ TimeSeriesDataset test failed: {e}")
        return False


def test_mitsui_dataset():
    """Test MitsuiDataset functionality."""
    print("Testing MitsuiDataset...")
    
    try:
        # Create mock Mitsui data files
        data, targets = create_mock_data()
        
        # Create temporary CSV files
        data_df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
        targets_df = pd.DataFrame(targets, columns=[f'target_{i}' for i in range(targets.shape[1])])
        
        # Add date_id column
        data_df['date_id'] = range(len(data_df))
        targets_df['date_id'] = range(len(targets_df))
        
        # Save temporary files
        data_path = '/tmp/test_train.csv'
        labels_path = '/tmp/test_train_labels.csv'
        
        data_df.to_csv(data_path, index=False)
        targets_df.to_csv(labels_path, index=False)
        
        # Test MitsuiDataset initialization
        dataset = MitsuiDataset(
            data_path=data_path,
            labels_path=labels_path,
            flag='train',
            size=(96, 48, 96),
            features='M',
            target='OT',
            scale=True,
            timeenc=1,
            freq='h',
            max_targets=20  # Limit for testing
        )
        
        print(f"✅ MitsuiDataset initialized: {len(dataset)} samples")
        
        # Test data loading
        sample = dataset[0]
        batch_x, batch_y, batch_x_mark, batch_y_mark = sample
        
        expected_x_shape = (96, 21)  # seq_len, n_features
        expected_y_shape = (144, 20)  # label_len + pred_len, max_targets
        
        if batch_x.shape == expected_x_shape and batch_y.shape == expected_y_shape:
            print(f"✅ MitsuiDataset sample shapes: x={batch_x.shape}, y={batch_y.shape}")
        else:
            print(f"❌ MitsuiDataset sample shapes incorrect: x={batch_x.shape}, y={batch_y.shape}")
            return False
        
        # Test temporal splitting
        train_dataset = MitsuiDataset(
            data_path=data_path,
            labels_path=labels_path,
            flag='train',
            size=(96, 48, 96),
            max_targets=20
        )
        
        val_dataset = MitsuiDataset(
            data_path=data_path,
            labels_path=labels_path,
            flag='val',
            size=(96, 48, 96),
            max_targets=20
        )
        
        test_dataset = MitsuiDataset(
            data_path=data_path,
            labels_path=labels_path,
            flag='test',
            size=(96, 48, 96),
            max_targets=20
        )
        
        print(f"✅ Temporal splitting: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        
        # Clean up temporary files
        os.remove(data_path)
        os.remove(labels_path)
        
        return True
        
    except Exception as e:
        print(f"❌ MitsuiDataset test failed: {e}")
        return False


def test_data_provider():
    """Test data provider functionality."""
    print("Testing data provider...")
    
    try:
        # Create mock arguments
        class MockArgs:
            def __init__(self):
                self.data_path = '/tmp/test_train.csv'
                self.labels_path = '/tmp/test_train_labels.csv'
                self.seq_len = 96
                self.label_len = 48
                self.pred_len = 96
                self.features = 'M'
                self.target = 'OT'
                self.scale = True
                self.timeenc = 1
                self.freq = 'h'
                self.batch_size = 16
                self.num_workers = 0
                self.max_targets = 20
        
        args = MockArgs()
        
        # Create mock data files
        data, targets = create_mock_data()
        data_df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
        targets_df = pd.DataFrame(targets, columns=[f'target_{i}' for i in range(targets.shape[1])])
        data_df['date_id'] = range(len(data_df))
        targets_df['date_id'] = range(len(targets_df))
        
        data_df.to_csv(args.data_path, index=False)
        targets_df.to_csv(args.labels_path, index=False)
        
        # Test data provider
        train_dataset, train_loader = mitsui_data_provider(args, 'train')
        val_dataset, val_loader = mitsui_data_provider(args, 'val')
        
        print(f"✅ Data provider: train={len(train_dataset)}, val={len(val_dataset)}")
        
        # Test data loader
        batch = next(iter(train_loader))
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        
        expected_batch_size = min(args.batch_size, len(train_dataset))
        expected_x_shape = (expected_batch_size, 96, 21)
        expected_y_shape = (expected_batch_size, 144, 20)
        
        if batch_x.shape == expected_x_shape and batch_y.shape == expected_y_shape:
            print(f"✅ DataLoader batch shapes: x={batch_x.shape}, y={batch_y.shape}")
        else:
            print(f"❌ DataLoader batch shapes incorrect: x={batch_x.shape}, y={batch_y.shape}")
            return False
        
        # Clean up
        os.remove(args.data_path)
        os.remove(args.labels_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Data provider test failed: {e}")
        return False


def test_preprocessing():
    """Test preprocessing functionality."""
    print("Testing preprocessing...")
    
    try:
        # Create mock data
        data, targets = create_mock_data()
        
        # Test TimeSeriesPreprocessor
        preprocessor = TimeSeriesPreprocessor(
            scale=True,
            timeenc=1,
            freq='h'
        )
        
        # Test preprocessing
        processed_data, processed_targets, time_features = preprocessor.fit_transform(data, targets)
        
        if processed_data.shape == data.shape and processed_targets.shape == targets.shape:
            print(f"✅ Preprocessing: data={processed_data.shape}, targets={processed_targets.shape}")
        else:
            print(f"❌ Preprocessing shape error")
            return False
        
        # Test inverse transform
        original_data = preprocessor.inverse_transform(processed_data, is_target=False)
        original_targets = preprocessor.inverse_transform(processed_targets, is_target=True)
        
        if np.allclose(original_data, data, atol=1e-6) and np.allclose(original_targets, targets, atol=1e-6):
            print("✅ Inverse transform works correctly")
        else:
            print("❌ Inverse transform error")
            return False
        
        # Test WindowGenerator
        window_gen = WindowGenerator(
            seq_len=96,
            label_len=48,
            pred_len=96
        )
        
        windows = window_gen.create_windows(processed_data, processed_targets)
        
        if len(windows) > 0:
            x, y = windows[0]
            if x.shape == (96, 21) and y.shape == (144, 50):
                print(f"✅ Window generation: {len(windows)} windows")
            else:
                print(f"❌ Window generation shape error: x={x.shape}, y={y.shape}")
                return False
        else:
            print("❌ No windows generated")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False


def test_time_features():
    """Test time feature generation."""
    print("Testing time features...")
    
    try:
        # Test time feature generation
        from utils.timefeatures import time_features
        
        # Create mock timestamps
        timestamps = pd.date_range('2023-01-01', periods=1000, freq='H')
        
        # Test different frequencies
        for freq in ['h', 'd', 'w', 'm']:
            features = time_features(timestamps, freq=freq)
            if features.shape[0] == len(timestamps):
                print(f"✅ Time features for {freq}: {features.shape}")
            else:
                print(f"❌ Time features for {freq} error: {features.shape}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Time features test failed: {e}")
        return False


def test_data_consistency():
    """Test data consistency across different operations."""
    print("Testing data consistency...")
    
    try:
        # Create mock data
        data, targets = create_mock_data()
        
        # Test with different flags
        train_dataset = TimeSeriesDataset(
            data=data, targets=targets, flag='train', size=(96, 48, 96), scale=True
        )
        
        val_dataset = TimeSeriesDataset(
            data=data, targets=targets, flag='val', size=(96, 48, 96), scale=True
        )
        
        # Test that scaling parameters are consistent
        if hasattr(train_dataset, 'scaler_mean_x') and hasattr(val_dataset, 'scaler_mean_x'):
            if np.allclose(train_dataset.scaler_mean_x, val_dataset.scaler_mean_x):
                print("✅ Scaling parameters consistent")
            else:
                print("❌ Scaling parameters inconsistent")
                return False
        
        # Test data loader consistency
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        if len(train_batch) == 4 and len(val_batch) == 4:
            print("✅ DataLoader consistency: 4 elements per batch")
        else:
            print("❌ DataLoader consistency error")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data consistency test failed: {e}")
        return False


def test_memory_efficiency():
    """Test memory efficiency with large datasets."""
    print("Testing memory efficiency...")
    
    try:
        # Create larger mock data
        n_samples = 5000
        n_features = 50
        n_targets = 100
        
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        targets = np.random.randn(n_samples, n_targets).astype(np.float32)
        
        # Test with limited targets
        dataset = TimeSeriesDataset(
            data=data,
            targets=targets,
            flag='train',
            size=(96, 48, 96),
            scale=True
        )
        
        # Test memory usage
        sample = dataset[0]
        batch_x, batch_y, batch_x_mark, batch_y_mark = sample
        
        # Check that memory usage is reasonable
        memory_mb = (batch_x.numel() + batch_y.numel()) * 4 / (1024 * 1024)  # 4 bytes per float32
        
        if memory_mb < 100:  # Less than 100MB per sample
            print(f"✅ Memory efficiency: {memory_mb:.2f}MB per sample")
        else:
            print(f"⚠️  High memory usage: {memory_mb:.2f}MB per sample")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory efficiency test failed: {e}")
        return False


def main():
    """Run all data processing tests."""
    print("=" * 60)
    print("Data Processing Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: TimeSeriesDataset
    test_results['timeseries_dataset'] = test_timeseries_dataset()
    
    # Test 2: MitsuiDataset
    test_results['mitsui_dataset'] = test_mitsui_dataset()
    
    # Test 3: Data provider
    test_results['data_provider'] = test_data_provider()
    
    # Test 4: Preprocessing
    test_results['preprocessing'] = test_preprocessing()
    
    # Test 5: Time features
    test_results['time_features'] = test_time_features()
    
    # Test 6: Data consistency
    test_results['data_consistency'] = test_data_consistency()
    
    # Test 7: Memory efficiency
    test_results['memory_efficiency'] = test_memory_efficiency()
    
    # Summary
    print("\n" + "=" * 60)
    print("Data Processing Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All data processing tests passed! Implementation is working correctly.")
        return 0
    else:
        print("⚠️  Some data processing tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
