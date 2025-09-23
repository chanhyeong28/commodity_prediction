"""
Chronos Time Series Embedding Example
====================================

This script demonstrates how to use Chronos (Amazon's time series model)
to generate embeddings for time series data, similar to how language models
generate embeddings for text.

Chronos adapts T5 language models for time series forecasting and embedding generation.

Installation:
    pip install git+https://github.com/amazon-science/chronos-forecasting.git

Usage:
    python chronos_embedding_example.py
"""

import pandas as pd
import torch
import numpy as np
from pathlib import Path

def install_chronos():
    """Install Chronos if not already installed."""
    try:
        from chronos import ChronosPipeline
        print("✅ Chronos is already installed")
        return True
    except ImportError:
        print("❌ Chronos not found. Installing...")
        import subprocess
        import sys
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/amazon-science/chronos-forecasting.git"
            ])
            print("✅ Chronos installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Chronos: {e}")
            return False

def read_time_series_data(file_path: str, column_name: str, start_id: int, end_id: int):
    """
    Read time series data from CSV file.
    
    Args:
        file_path: Path to CSV file
        column_name: Name of the column to extract
        start_id: Starting date_id
        end_id: Ending date_id
    
    Returns:
        numpy array of time series values
    """
    print(f"Reading {column_name} from date_id {start_id} to {end_id}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Filter by date_id range
    mask = (df['date_id'] >= start_id) & (df['date_id'] <= end_id)
    filtered_df = df[mask]
    
    if filtered_df.empty:
        raise ValueError(f"No data found for date_id range {start_id}-{end_id}")
    
    # Extract the time series
    time_series = filtered_df[column_name].values
    
    # Handle missing values
    if pd.isna(time_series).any():
        print("Warning: Found missing values, forward filling...")
        time_series = pd.Series(time_series).fillna(method='ffill').values
    
    print(f"Extracted time series with {len(time_series)} points")
    print(f"Values: {time_series}")
    print(f"Mean: {np.mean(time_series):.4f}, Std: {np.std(time_series):.4f}")
    
    return time_series

def create_chronos_embeddings(time_series: np.ndarray, model_name: str = "amazon/chronos-t5-small"):
    """
    Create embeddings using Chronos model.
    
    Args:
        time_series: Input time series data
        model_name: Name of the Chronos model to use
    
    Returns:
        Embedding tensor
    """
    try:
        from chronos import ChronosPipeline
    except ImportError:
        print("❌ Chronos not available. Please install it first.")
        return None
    
    print(f"Creating embeddings using {model_name}")
    
    # Initialize the Chronos pipeline
    try:
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        print(f"✅ Loaded {model_name} successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Convert time series to tensor
    context = torch.tensor(time_series, dtype=torch.float32)
    
    # Generate embeddings
    try:
        embeddings, tokenizer_state = pipeline.embed(context)
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        return None

def main():
    """Main function to demonstrate Chronos embeddings."""
    print("Chronos Time Series Embedding Example")
    print("=" * 50)
    
    # Check if Chronos is installed
    if not install_chronos():
        print("Please install Chronos manually:")
        print("pip install git+https://github.com/amazon-science/chronos-forecasting.git")
        return
    
    # File paths
    train_file = "/Users/minkeychang/commodity_prediction/kaggle/train.csv"
    
    # Check if file exists
    if not Path(train_file).exists():
        print(f"❌ Error: File {train_file} not found!")
        print("Please make sure the file path is correct.")
        return
    
    try:
        # Read LME_AH_Close from date_id 26 to 40
        time_series = read_time_series_data(
            file_path=train_file,
            column_name="LME_AH_Close",
            start_id=26,
            end_id=40
        )
        
        print("\n" + "="*50)
        print("Creating Chronos Embeddings")
        print("="*50)
        
        # Try different model sizes
        model_sizes = [
            "amazon/chronos-t5-tiny",    # Smallest, fastest
            "amazon/chronos-t5-small",   # Good balance
            # "amazon/chronos-t5-base",  # Larger, more accurate
            # "amazon/chronos-t5-large", # Largest, most accurate
        ]
        
        embeddings_results = {}
        
        for model_name in model_sizes:
            print(f"\n--- Using {model_name} ---")
            embeddings = create_chronos_embeddings(time_series, model_name)
            
            if embeddings is not None:
                embeddings_results[model_name] = embeddings
                
                # Show embedding statistics
                print(f"Embedding shape: {embeddings.shape}")
                print(f"Embedding mean: {embeddings.mean().item():.4f}")
                print(f"Embedding std: {embeddings.std().item():.4f}")
                print(f"Embedding min: {embeddings.min().item():.4f}")
                print(f"Embedding max: {embeddings.max().item():.4f}")
                
                # Show first few values
                if len(embeddings.shape) == 1:
                    print(f"First 10 values: {embeddings[:10].tolist()}")
                else:
                    print(f"First row (first 10 values): {embeddings[0, :10].tolist()}")
        
        # Save embeddings
        if embeddings_results:
            output_file = "/Users/minkeychang/commodity_prediction/chronos_embeddings.npz"
            
            # Convert tensors to numpy arrays for saving
            save_data = {
                'time_series': time_series,
            }
            
            for model_name, embeddings in embeddings_results.items():
                # Convert model name to valid key
                key = model_name.replace('amazon/', '').replace('-', '_')
                save_data[key] = embeddings.cpu().numpy()
            
            np.savez(output_file, **save_data)
            print(f"\n✅ Embeddings saved to: {output_file}")
            
            # Load and verify
            loaded_data = np.load(output_file)
            print(f"Loaded embeddings from file:")
            for key in loaded_data.keys():
                print(f"  {key}: {loaded_data[key].shape}")
        
        print("\n" + "="*50)
        print("Summary")
        print("="*50)
        print(f"Time series length: {len(time_series)}")
        print(f"Number of models used: {len(embeddings_results)}")
        print(f"Embeddings saved to: {output_file if embeddings_results else 'None'}")
        
        if embeddings_results:
            print("\nAvailable embedding models:")
            for model_name in embeddings_results.keys():
                print(f"  ✅ {model_name}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
