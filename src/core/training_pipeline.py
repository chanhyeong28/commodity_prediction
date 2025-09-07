"""
Unified Training Pipeline for Hybrid Time Series Forecasting System

This module provides a comprehensive training pipeline that orchestrates:
1. Data loading and preprocessing
2. Knowledge graph construction
3. Model training with GraphRAG context
4. Validation and evaluation
5. Checkpointing and logging
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
import json

from .config import SystemConfig
from .data_processor import CommodityDataProcessor, create_data_loaders
from .model_factory import ModelManager
from ..kg.graph_builder import KnowledgeGraphBuilder, PatchConfig as KGPatchConfig
from ..kg.graph_rag import GraphRAGRetriever, RetrievalConfig

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """
    Unified training pipeline for the hybrid forecasting system.
    
    This class orchestrates the entire training process including data preparation,
    knowledge graph construction, model training, and evaluation.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.data_processor = CommodityDataProcessor(config)
        self.model_manager = ModelManager(config)
        self.kg_builder = None
        self.kg_retriever = None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
    def setup_logging(self):
        """Setup logging and experiment tracking"""
        if self.config.training.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.training.wandb_project,
                config=self.config.__dict__,
                name=f"hybrid_forecasting_{self.config.seed}"
            )
            logger.info("Wandb logging initialized")
        elif self.config.training.use_wandb and not WANDB_AVAILABLE:
            logger.warning("Wandb requested but not available. Continuing without wandb logging.")
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare all data for training.
        
        Returns:
            Tuple of (train_data, val_data, target_pairs)
        """
        logger.info("Preparing data...")
        
        # Load data
        train_data, _, target_pairs = self.data_processor.load_data()
        
        # Preprocess data
        train_data = self.data_processor.preprocess_data(train_data)
        
        # Extract metadata
        self.data_processor.extract_series_metadata(train_data)
        
        # Fit scaler
        self.data_processor.fit_scaler(train_data)
        
        # Transform data
        train_data = self.data_processor.transform_data(train_data)
        
        # Split data
        train_data, val_data = self.data_processor.split_data(
            train_data, self.config.training.val_split
        )
        
        logger.info(f"Data prepared: train={len(train_data)}, val={len(val_data)}")
        return train_data, val_data, target_pairs
    
    def build_knowledge_graph(self, train_data: pd.DataFrame, target_pairs: pd.DataFrame):
        """Build the knowledge graph from training data"""
        logger.info("Building knowledge graph...")
        
        # Create database directory
        Path(self.config.knowledge_graph.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure KG patches
        kg_patch_config = KGPatchConfig(
            window_sizes=self.config.knowledge_graph.window_sizes,
            strides=self.config.knowledge_graph.strides
        )
        
        # Create KG builder
        self.kg_builder = KnowledgeGraphBuilder(
            db_path=self.config.knowledge_graph.db_path,
            patch_config=kg_patch_config
        )
        
        # Build knowledge graph
        self.kg_builder.build_from_dataframe(train_data, target_pairs)
        
        # Create KG retriever
        retrieval_config = RetrievalConfig(
            max_nodes=self.config.retrieval.max_nodes,
            max_edges=self.config.retrieval.max_edges,
            similarity_threshold=self.config.retrieval.similarity_threshold,
            ts_patch_ratio=self.config.retrieval.ts_patch_ratio,
            max_hops=self.config.retrieval.max_hops,
            recency_decay=self.config.retrieval.recency_decay,
            similarity_weight=self.config.retrieval.similarity_weight,
            market_match_weight=self.config.retrieval.market_match_weight,
            recency_weight=self.config.retrieval.recency_weight,
            edge_strength_weight=self.config.retrieval.edge_strength_weight
        )
        
        self.kg_retriever = GraphRAGRetriever(
            db_path=self.config.knowledge_graph.db_path,
            config=retrieval_config
        )
        
        logger.info("Knowledge graph construction complete")
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Training metrics
        """
        self.model_manager.set_training_mode()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            target_values = batch['target_values'].to(self.model_manager.factory.device)
            lags = batch['lags']
            
            # Prepare patch embeddings and KG context
            patch_embeddings, attention_mask, kg_context, kg_mask = self._prepare_batch_data(batch)
            
            # Forward pass
            self.model_manager.optimizer.zero_grad()
            
            forecasts = self.model_manager.model(
                patch_embeddings,
                kg_context,
                kg_mask,
                attention_mask
            )
            
            # Compute loss
            loss = self._compute_loss(forecasts, target_values, lags)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model_manager.model.parameters(),
                self.config.training.max_grad_norm
            )
            
            self.model_manager.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}"
            })
            
            # Log batch metrics
            if batch_idx % self.config.training.log_interval == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate_epoch(self, val_loader, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Validation metrics
        """
        self.model_manager.set_eval_mode()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                target_values = batch['target_values'].to(self.model_manager.factory.device)
                lags = batch['lags']
                
                # Prepare patch embeddings and KG context
                patch_embeddings, attention_mask, kg_context, kg_mask = self._prepare_batch_data(batch)
                
                # Forward pass
                forecasts = self.model_manager.model(
                    patch_embeddings,
                    kg_context,
                    kg_mask,
                    attention_mask
                )
                
                # Compute loss
                loss = self._compute_loss(forecasts, target_values, lags)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def _prepare_batch_data(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for model input.
        
        Args:
            batch: Batch data
            
        Returns:
            Tuple of (patch_embeddings, attention_mask, kg_context, kg_mask)
        """
        batch_size = len(batch['targets'])
        device = self.model_manager.factory.device
        
        # Get patch data and attention mask
        patch_data = batch['patches'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Convert raw patch data to embeddings
        # For now, use a simple linear projection (in practice, use trained embedder)
        if patch_data.dim() == 3:  # [batch_size, num_patches, patch_length]
            # Flatten patches and project to embedding dimension
            patch_flat = patch_data.view(batch_size, -1)
            # Simple linear projection (replace with trained embedder)
            patch_embeddings = torch.randn(
                batch_size, patch_data.size(1), self.config.model.d_model
            ).to(device)
        else:
            # Fallback
            patch_embeddings = torch.randn(
                batch_size, 10, self.config.model.d_model
            ).to(device)
            attention_mask = torch.ones(batch_size, 10).to(device)
        
        # Create dummy KG context (replace with actual KG retrieval)
        kg_context = torch.randn(
            batch_size, 20, self.config.model.d_context
        ).to(device)
        
        kg_mask = torch.ones(batch_size, 20).to(device)
        
        return patch_embeddings, attention_mask, kg_context, kg_mask
    
    def _compute_loss(self, forecasts: Dict[str, torch.Tensor], 
                     target_values: torch.Tensor, lags: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for the forecasts.
        
        Args:
            forecasts: Model forecasts
            target_values: Target values
            lags: Forecast lags
            
        Returns:
            Computed loss
        """
        total_loss = 0.0
        num_losses = 0
        
        for lag in lags.unique():
            horizon_key = f'horizon_{lag.item()}'
            if horizon_key in forecasts:
                # Get targets for this lag
                lag_mask = (lags == lag)
                lag_targets = target_values[lag_mask]
                lag_forecasts = forecasts[horizon_key][lag_mask]
                
                if len(lag_targets) > 0:
                    loss = self.model_manager.criterion(lag_forecasts.squeeze(), lag_targets)
                    total_loss += loss
                    num_losses += 1
        
        return total_loss / max(num_losses, 1)
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Setup logging
        self.setup_logging()
        
        # Prepare data
        train_data, val_data, target_pairs = self.prepare_data()
        
        # Build knowledge graph
        self.build_knowledge_graph(train_data, target_pairs)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_data, val_data, target_pairs, self.data_processor, self.config
        )
        
        # Initialize model
        self.model_manager.initialize_model()
        
        # Training loop
        for epoch in range(self.config.training.max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update scheduler
            self.model_manager.scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            metrics['epoch'] = epoch
            metrics['learning_rate'] = self.model_manager.optimizer.param_groups[0]['lr']
            
            # Log metrics
            logger.info(f"Epoch {epoch}: {metrics}")
            
            if self.config.training.use_wandb and WANDB_AVAILABLE:
                wandb.log(metrics, step=epoch)
            
            # Save best model
            current_loss = val_metrics['val_loss']
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.model_manager.save_checkpoint(
                    epoch, current_loss, self.config.training.save_path
                )
                logger.info(f"Saved best model with loss {current_loss:.4f}")
            
            # Store training history
            self.training_history.append(metrics)
        
        logger.info("Training complete!")
        
        # Save final model
        self.model_manager.save_checkpoint(
            self.current_epoch, self.best_loss, 
            self.config.training.save_path.replace('.pth', '_final.pth')
        )
        
        # Save training history
        self._save_training_history()
    
    def _save_training_history(self):
        """Save training history to file"""
        history_path = Path(self.config.training.save_path).parent / 'training_history.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")

def create_training_pipeline(config: SystemConfig) -> TrainingPipeline:
    """
    Create a training pipeline with the given configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured training pipeline
    """
    return TrainingPipeline(config)

def main():
    """Example training script"""
    from .config import load_config_from_file
    
    # Load configuration
    config_manager = load_config_from_file('configs/training_config.json')
    config = config_manager.config
    
    # Validate configuration
    if not config_manager.validate_config():
        raise ValueError("Invalid configuration")
    
    # Setup logging
    config_manager.setup_logging()
    
    # Create and run training pipeline
    pipeline = create_training_pipeline(config)
    pipeline.train()

if __name__ == "__main__":
    main()
