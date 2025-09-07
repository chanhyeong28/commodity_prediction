#!/usr/bin/env python3
"""
Unified Main Script for Hybrid Time Series Forecasting System

This script provides a clean, unified interface for:
1. Building the knowledge graph
2. Training the Time-LlaMA model
3. Running inference for Kaggle submission
4. Complete end-to-end pipeline

Usage:
    python main.py --mode train --config configs/training_config.json
    python main.py --mode inference --config configs/inference_config.json
    python main.py --mode build_kg --config configs/training_config.json
    python main.py --mode full_pipeline --config configs/training_config.json
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config import load_config_from_file, SystemConfig
from src.core.training_pipeline import create_training_pipeline
from src.core.inference_pipeline import create_inference_pipeline
from src.core.data_processor import CommodityDataProcessor
from src.kg.graph_builder import KnowledgeGraphBuilder, PatchConfig as KGPatchConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_knowledge_graph(config: SystemConfig):
    """Build the knowledge graph from training data"""
    logger.info("Building knowledge graph...")
    
    # Create data processor
    data_processor = CommodityDataProcessor(config)
    
    # Load data
    train_data, _, target_pairs = data_processor.load_data()
    
    # Preprocess data
    train_data = data_processor.preprocess_data(train_data)
    
    # Create database directory
    Path(config.knowledge_graph.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure KG patches
    kg_patch_config = KGPatchConfig(
        window_sizes=config.knowledge_graph.window_sizes,
        strides=config.knowledge_graph.strides
    )
    
    # Build knowledge graph
    builder = KnowledgeGraphBuilder(
        db_path=config.knowledge_graph.db_path,
        patch_config=kg_patch_config
    )
    
    builder.build_from_dataframe(train_data, target_pairs)
    
    logger.info("Knowledge graph construction complete!")

def train_model(config: SystemConfig):
    """Train the hybrid forecasting model"""
    logger.info("Starting model training...")
    
    # Create training pipeline
    pipeline = create_training_pipeline(config)
    
    # Run training
    pipeline.train()
    
    logger.info("Model training complete!")

def run_inference(config: SystemConfig):
    """Run inference for Kaggle submission"""
    logger.info("Starting inference...")
    
    # Create inference pipeline
    pipeline = create_inference_pipeline(config)
    
    # Load model
    pipeline.load_model()
    
    # Setup knowledge graph
    pipeline.setup_knowledge_graph()
    
    # Load test data
    test_data = pd.read_csv(config.test_data_path)
    all_data = pd.read_csv(config.train_data_path)
    
    # Run inference
    predictions = pipeline.run_inference(test_data, all_data)
    
    # Save predictions
    pipeline.save_predictions(predictions)
    
    logger.info("Inference complete!")
    return predictions

def run_full_pipeline(config: SystemConfig):
    """Run the complete end-to-end pipeline"""
    logger.info("Starting full pipeline...")
    
    # Step 1: Build knowledge graph
    build_knowledge_graph(config)
    
    # Step 2: Train model
    train_model(config)
    
    # Step 3: Run inference
    predictions = run_inference(config)
    
    logger.info("Full pipeline complete!")
    return predictions

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Hybrid Time Series Forecasting System')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['build_kg', 'train', 'inference', 'full_pipeline'],
                       help='Mode to run: build_kg, train, inference, or full_pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path('database').mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config_manager = load_config_from_file(args.config)
    config = config_manager.config
    
    # Validate configuration
    if not config_manager.validate_config():
        logger.error("Invalid configuration")
        return
    
    # Setup logging
    config_manager.setup_logging()
    
    # Update output paths
    config.training.save_path = str(Path(args.output_dir) / 'best_model.pth')
    config.inference.output_path = str(Path(args.output_dir) / 'submission.csv')
    
    # Run the requested mode
    if args.mode == 'build_kg':
        build_knowledge_graph(config)
        
    elif args.mode == 'train':
        train_model(config)
        
    elif args.mode == 'inference':
        predictions = run_inference(config)
        print(f"Generated predictions for {len(predictions)} rows")
        
    elif args.mode == 'full_pipeline':
        predictions = run_full_pipeline(config)
        print(f"Generated predictions for {len(predictions)} rows")

if __name__ == "__main__":
    main()
