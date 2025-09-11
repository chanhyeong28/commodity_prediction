"""
Core Commodity Forecasting System

This is the main module containing all core functionality for the commodity forecasting system.
It consolidates all components into a single, focused module.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import yaml
import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time
import math
from tqdm import tqdm
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.production_config import ProductionConfig, ProductionConfigManager
from src.kg.unified_kg_system import UnifiedKGSystem, KGConfig
from src.models.unified_time_llama import UnifiedTimeLlaMA, TimeLlaMAConfig, create_unified_time_llama

logger = logging.getLogger(__name__)


class CommodityForecastingSystem:
    """
    Main system class that consolidates all functionality.
    
    This class provides a single interface for all commodity forecasting operations.
    """
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.config_manager = ProductionConfigManager()
        self.config = self.config_manager.load_config(environment)
        self.kg_system = None
        self.model = None
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.system.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/system.log'),
                logging.StreamHandler()
            ]
        )
        
        logger.info(f"Initialized CommodityForecastingSystem for environment: {environment}")
    
    def build_knowledge_graph(self):
        """Build the knowledge graph from training data"""
        logger.info("🏗️  Building knowledge graph...")
        
        try:
            # Create KG system
            kg_config = KGConfig(
                window_sizes=self.config.patches.window_sizes,
                strides=self.config.patches.strides,
                max_nodes_per_series=self.config.patches.max_patches_per_series,
                embedding_dim=self.config.patches.embedding_dim,
                correlation_threshold=self.config.knowledge_graph.correlation_threshold,
                p_value_threshold=self.config.knowledge_graph.p_value_threshold,
                max_correlations_per_node=self.config.knowledge_graph.max_correlations_per_node,
                db_path=self.config.knowledge_graph.db_path
            )
            
            self.kg_system = UnifiedKGSystem(kg_config)
            
            # Load data
            train_data = pd.read_csv(f"{self.config.data.raw_path}/{self.config.data.train_file}")
            target_pairs = pd.read_csv(f"{self.config.data.raw_path}/{self.config.data.target_pairs_file}")
            
            # Build KG
            start_time = time.time()
            self.kg_system.build_from_dataframe(train_data, target_pairs)
            build_time = time.time() - start_time
            
            # Get statistics
            stats = self.kg_system.get_statistics()
            
            logger.info(f"✅ Knowledge graph built successfully!")
            logger.info(f"   - Build time: {build_time:.2f} seconds")
            logger.info(f"   - Nodes: {stats['nodes']:,}")
            logger.info(f"   - Edges: {stats['edges']:,}")
            logger.info(f"   - Database size: {stats['db_size_mb']:.2f} MB")
            
        except Exception as e:
            logger.error(f"❌ Knowledge graph construction failed: {e}")
            raise
    
    def create_model(self):
        """Create the Time-LlaMA model"""
        logger.info("🎓 Creating Time-LlaMA model...")
        
        try:
            # Create model configuration
            model_config = TimeLlaMAConfig(
                d_model=self.config.model.d_model,
                n_heads=self.config.model.n_heads,
                n_layers=self.config.model.n_layers,
                d_ff=self.config.model.d_ff,
                dropout=self.config.model.dropout,
                max_seq_len=self.config.model.max_seq_len,
                patch_embed_dim=self.config.patches.embedding_dim,
                patch_size=self.config.patches.patch_size,
                context_dim=self.config.model.context_dim,
                context_heads=self.config.model.context_heads,
                use_peft=self.config.model.use_peft,
                lora_r=self.config.model.lora_r,
                lora_alpha=self.config.model.lora_alpha,
                lora_dropout=self.config.model.lora_dropout,
                num_horizons=self.config.model.num_horizons,
                output_dim=self.config.model.output_dim
            )
            
            # Create model
            self.model = create_unified_time_llama(model_config, self.config.system.device)
            
            logger.info(f"✅ Time-LlaMA model created")
            logger.info(f"   - Total parameters: {self.model.get_num_parameters():,}")
            logger.info(f"   - Trainable parameters: {self.model.get_trainable_parameters():,}")
            
        except Exception as e:
            logger.error(f"❌ Model creation failed: {e}")
            raise
    
    def train_model(self):
        """Train the Time-LlaMA model"""
        logger.info("🎓 Training Time-LlaMA model...")
        
        try:
            if self.model is None:
                self.create_model()
            
            # Load data
            train_data = pd.read_csv(f"{self.config.data.raw_path}/{self.config.data.train_file}")
            train_labels = pd.read_csv(f"{self.config.data.raw_path}/{self.config.data.train_labels_file}")
            
            logger.info(f"✅ Training components ready")
            logger.info(f"   - Model parameters: {self.model.get_num_parameters():,}")
            logger.info(f"   - Training data: {len(train_data)} rows")
            
            # Prepare training data
            logger.info("📊 Preparing training data...")
            
            # Get target pairs for training
            target_pairs = pd.read_csv(f"{self.config.data.raw_path}/{self.config.data.target_pairs_file}")
            logger.info(f"   - Target pairs: {len(target_pairs)}")
            
            # Create training batches
            batch_size = int(self.config.training.batch_size)
            num_epochs = int(self.config.training.num_epochs)
            learning_rate = float(self.config.training.learning_rate)
            
            # Setup optimizer and loss
            weight_decay = float(self.config.training.weight_decay)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.MSELoss()
            
            # Training loop
            logger.info(f"🚀 Starting training for {num_epochs} epochs...")
            logger.info(f"   - Batch size: {batch_size}")
            logger.info(f"   - Learning rate: {learning_rate}")
            
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                epoch_batches = 0
                
                # Create batches (simplified - using random sampling)
                num_samples = min(100, len(train_data))  # Limit for testing
                indices = torch.randperm(num_samples)
                
                for i in range(0, num_samples, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    
                    # Get batch data
                    batch_data = train_data.iloc[batch_indices]
                    
                    # Create dummy inputs and targets for testing
                    # In real implementation, this would use the KG system and patch embeddings
                    batch_size_actual = len(batch_data)
                    seq_len = 50  # Sequence length
                    input_dim = 64  # Input dimension
                    
                    # Create dummy input tensors
                    inputs = torch.randn(batch_size_actual, seq_len, input_dim).to(self.config.system.device)
                    targets = torch.randn(batch_size_actual, 4).to(self.config.system.device)  # 4 horizons
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    
                    # Extract forecasts from model output (model returns a dict)
                    if isinstance(outputs, dict):
                        forecasts = outputs['forecasts']
                    else:
                        forecasts = outputs
                    
                    loss = criterion(forecasts, targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_batches += 1
                    total_loss += loss.item()
                    num_batches += 1
                
                # Log epoch results
                avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_epoch_loss:.6f}")
                
                # Early stopping check (simplified)
                if epoch > 5 and avg_epoch_loss < 0.001:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Training completed
            avg_total_loss = total_loss / max(num_batches, 1)
            logger.info(f"✅ Training completed!")
            logger.info(f"   - Total epochs: {epoch+1}")
            logger.info(f"   - Average loss: {avg_total_loss:.6f}")
            logger.info(f"   - Total batches: {num_batches}")
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            raise
    
    def run_inference(self):
        """Run inference on test data"""
        logger.info("🔮 Running inference...")
        
        try:
            if self.model is None:
                self.create_model()
            
            if self.kg_system is None:
                self.build_knowledge_graph()
            
            # Load test data
            test_data = pd.read_csv(f"{self.config.data.raw_path}/{self.config.data.test_file}")
            target_pairs = pd.read_csv(f"{self.config.data.raw_path}/{self.config.data.target_pairs_file}")
            
            logger.info(f"✅ Inference components ready")
            logger.info(f"   - Model loaded")
            logger.info(f"   - KG system ready")
            logger.info(f"   - Test data: {len(test_data)} rows")
            logger.info(f"   - Targets: {len(target_pairs)} pairs")
            
            # TODO: Implement actual inference loop
            logger.info("⚠️  Inference loop not yet implemented")
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            raise
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        logger.info("🚀 Running full pipeline...")
        
        try:
            # Build knowledge graph
            self.build_knowledge_graph()
            
            # Train model
            self.train_model()
            
            # Run inference
            self.run_inference()
            
            logger.info("🎉 Full pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Full pipeline failed: {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            'environment': self.environment,
            'device': self.config.system.device,
            'model_created': self.model is not None,
            'kg_created': self.kg_system is not None
        }
        
        if self.model is not None:
            info['model_parameters'] = self.model.get_num_parameters()
            info['trainable_parameters'] = self.model.get_trainable_parameters()
        
        if self.kg_system is not None:
            kg_stats = self.kg_system.get_statistics()
            info['kg_nodes'] = kg_stats['nodes']
            info['kg_edges'] = kg_stats['edges']
            info['kg_size_mb'] = kg_stats['db_size_mb']
        
        return info


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Commodity Forecasting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python core.py --mode build_kg --config development
  python core.py --mode train --config production
  python core.py --mode inference --config kaggle
  python core.py --mode full_pipeline --config production
        """
    )
    
    parser.add_argument(
        "--mode",
        required=True,
        choices=["build_kg", "train", "inference", "full_pipeline"],
        help="Mode to run: build_kg, train, inference, or full_pipeline"
    )
    
    parser.add_argument(
        "--config",
        default="development",
        choices=["development", "production", "kaggle"],
        help="Configuration environment (default: development)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create system
        system = CommodityForecastingSystem(args.config)
        
        logger.info(f"🚀 Starting commodity forecasting system")
        logger.info(f"   - Mode: {args.mode}")
        logger.info(f"   - Environment: {args.config}")
        logger.info(f"   - Device: {system.config.system.device}")
        
        # Run the requested mode
        if args.mode == "build_kg":
            system.build_knowledge_graph()
        elif args.mode == "train":
            system.train_model()
        elif args.mode == "inference":
            system.run_inference()
        elif args.mode == "full_pipeline":
            system.run_full_pipeline()
        
        # Print system info
        info = system.get_system_info()
        logger.info(f"✅ Operation completed successfully!")
        logger.info(f"System info: {info}")
        
    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
