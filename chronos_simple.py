"""
Simple Chronos Embedding Example
===============================

Quick example showing how to use Chronos to generate embeddings
for your LME_AH_Close time series data.

Installation:
    pip install git+https://github.com/amazon-science/chronos-forecasting.git
"""

import pandas as pd
import torch
import numpy as np

def main():
    print("Chronos Simple Embedding Example")
    print("=" * 40)
    
    # Read your data
    df = pd.read_csv("/Users/minkeychang/commodity_prediction/kaggle/train.csv")
    
    # Extract LME_AH_Close from date_id 26 to 40
    mask = (df['date_id'] >= 26) & (df['date_id'] <= 40)
    time_series = df[mask]['LME_AH_Close'].values
    
    print(f"Time series length: {len(time_series)}")
    print(f"Values: {time_series}")
    
    # Import Chronos
    try:
        from chronos import ChronosPipeline
        print("✅ Chronos imported successfully")
    except ImportError:
        print("❌ Chronos not installed. Run:")
        print("pip install git+https://github.com/amazon-science/chronos-forecasting.git")
        return
    
    # Load Chronos model
    try:
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",  # Small model for quick testing
            device_map="cpu",  # Use CPU for simplicity
            torch_dtype=torch.float32,
        )
        print("✅ Chronos model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Convert to tensor
    context = torch.tensor(time_series, dtype=torch.float32)
    
    # Generate embeddings
    try:
        embeddings, _ = pipeline.embed(context)
        print(f"✅ Embeddings generated!")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding type: {type(embeddings)}")
        
        # Show some statistics
        print(f"Mean: {embeddings.mean().item():.4f}")
        print(f"Std: {embeddings.std().item():.4f}")
        
        # Show first few values
        if len(embeddings.shape) == 1:
            print(f"First 10 values: {embeddings[:10].tolist()}")
        else:
            print(f"First row: {embeddings[0, :10].tolist()}")
        
        # Save embeddings
        np.save("/Users/minkeychang/commodity_prediction/chronos_embeddings.npy", 
                embeddings.cpu().numpy())
        print("✅ Embeddings saved to chronos_embeddings.npy")
        
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")

if __name__ == "__main__":
    main()
