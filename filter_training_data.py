#!/usr/bin/env python3
"""
Filter training data to remove overlapping test period (date_id >= 1827)
This ensures proper local testing without data leakage.
"""

import pandas as pd
import os

def filter_training_data():
    """Filter out test period from training data files."""
    
    # Define file paths
    train_file = "raw_data/train.csv"
    train_labels_file = "raw_data/train_labels.csv"
    
    # Check if files exist
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found")
        return
    
    if not os.path.exists(train_labels_file):
        print(f"Error: {train_labels_file} not found")
        return
    
    print("Loading training data...")
    
    # Load the data
    train_df = pd.read_csv(train_file)
    train_labels_df = pd.read_csv(train_labels_file)
    
    print(f"Original train.csv: {len(train_df)} rows")
    print(f"Original train_labels.csv: {len(train_labels_df)} rows")
    print(f"Date range in train.csv: {train_df['date_id'].min()} to {train_df['date_id'].max()}")
    print(f"Date range in train_labels.csv: {train_labels_df['date_id'].min()} to {train_labels_df['date_id'].max()}")
    
    # Filter out rows with date_id >= 1827 (test period starts at 1827)
    train_filtered = train_df[train_df['date_id'] < 1827].copy()
    train_labels_filtered = train_labels_df[train_labels_df['date_id'] < 1827].copy()
    
    print(f"\nAfter filtering (date_id < 1827):")
    print(f"Filtered train.csv: {len(train_filtered)} rows")
    print(f"Filtered train_labels.csv: {len(train_labels_filtered)} rows")
    print(f"Date range in filtered train.csv: {train_filtered['date_id'].min()} to {train_filtered['date_id'].max()}")
    print(f"Date range in filtered train_labels.csv: {train_labels_filtered['date_id'].min()} to {train_labels_filtered['date_id'].max()}")
    
    # Create backup of original files
    print("\nCreating backups...")
    train_backup = "raw_data/train_original.csv"
    train_labels_backup = "raw_data/train_labels_original.csv"
    
    if not os.path.exists(train_backup):
        train_df.to_csv(train_backup, index=False)
        print(f"Backup created: {train_backup}")
    else:
        print(f"Backup already exists: {train_backup}")
    
    if not os.path.exists(train_labels_backup):
        train_labels_df.to_csv(train_labels_backup, index=False)
        print(f"Backup created: {train_labels_backup}")
    else:
        print(f"Backup already exists: {train_labels_backup}")
    
    # Overwrite original files with filtered data
    print("\nOverwriting original files with filtered data...")
    train_filtered.to_csv(train_file, index=False)
    train_labels_filtered.to_csv(train_labels_file, index=False)
    
    print("✅ Training data filtering completed successfully!")
    print(f"Removed {len(train_df) - len(train_filtered)} rows from train.csv")
    print(f"Removed {len(train_labels_df) - len(train_labels_filtered)} rows from train_labels.csv")
    print(f"Test period (date_id >= 1827) has been excluded from training data")

if __name__ == "__main__":
    filter_training_data()
