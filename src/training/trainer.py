"""
Training Pipeline for Hybrid Time Series Forecasting System

This module implements the training pipeline that combines:
1. Knowledge graph construction and retrieval
2. Patch-based embedding
3. Time-LlaMA adapter with cross-attention
4. PEFT/LoRA for efficient training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from pathlib import Path
import wandb
from tqdm import tqdm
import pickle

# Import our modules
from ..kg.graph_builder import KnowledgeGraphBuilder, PatchConfig as KGPatchConfig
from ..kg.graph_rag import GraphRAGRetriever, RetrievalConfig
from ..data.patch_embedder import PatchEmbeddingPipeline, PatchConfig
from ..models.time_llama_adapter import TimeLlaMAWithPEFT, TimeLlaMAConfig

logger = logging.getLogger(__name__)

class CommodityDataset(Dataset):
    """
    Dataset for commodity time series forecasting.
    
    This dataset handles:
    1. Loading time series data
    2. Creating patches and embeddings
    3. Retrieving KG context
    4. Preparing targets for different horizons
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 target_pairs: pd.DataFrame,
                 patch_pipeline: PatchEmbeddingPipeline,
                 kg_retriever: GraphRAGRetriever,
                 config: Dict[str, Any]):
        self.data = data
        self.target_pairs = target_pairs
        self.patch_pipeline = patch_pipeline
        self.kg_retriever = kg_retriever
        self.config = config
        
        # Create training samples
        self.samples = self._create_samples()
        
    def _create_samples(self) -> List[Dict[str, Any]]:
        """Create training samples from the data"""
        samples = []
        
        # Get unique dates for training
        dates = sorted(self.data['date_id'].unique())
        
        # Create samples for each target pair
        for _, target_row in self.target_pairs.iterrows():
            target = target_row['target']
            pair = target_row['pair']
            lag = target_row['lag']
            
            # Parse target series
            target_series = self._parse_target(target)
            
            # Create samples for different forecast dates
            for i in range(self.config['min_lookback'], len(dates) - lag):
                forecast_date = dates[i + lag]
                lookback_date = dates[i]
                
                # Get lookback data
                lookback_data = self.data[self.data['date_id'] <= lookback_date]
                
                if len(lookback_data) < self.config['min_lookback_days']:
                    continue
                
                # Get target value
                target_data = self.data[self.data['date_id'] == forecast_date]
                if target_data.empty:
                    continue
                
                target_value = target_data[target].iloc[0] if target in target_data.columns else None
                if target_value is None or pd.isna(target_value):
                    continue
                
                sample = {
                    'target': target,
                    'pair': pair,
                    'lag': lag,
                    'forecast_date': forecast_date,
                    'lookback_date': lookback_date,
                    'target_value': target_value,
                    'lookback_data': lookback_data
                }
                
                samples.append(sample)
        
        logger.info(f"Created {len(samples)} training samples")
        return samples
    
    def _parse_target(self, target: str) -> str:
        """Parse target string to get series identifier"""
        # Handle different target formats
        if ' - ' in target:
            # Difference target: "LME_AL_Close - US_Stock_VT_adj_close"
            parts = target.split(' - ')
            return parts[0].strip()  # Use first series as primary
        else:
            return target
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample"""
        sample = self.samples[idx]
        
        # Get patch embeddings
        patch_embeddings = self.patch_pipeline.transform(
            sample['lookback_data'],
            target_series=[sample['target']]
        )
        
        if sample['target'] not in patch_embeddings:
            # Fallback: create dummy embeddings
            patch_embeddings = {
                sample['target']: {
                    'embeddings': torch.zeros(1, 10, self.patch_pipeline.get_embedding_dim()),
                    'attention_mask': torch.ones(1, 10),
                    'patches': []
                }
            }
        
        # Get KG context
        series_data = sample['lookback_data'][sample['target']].dropna().tail(30)
        kg_context = self.kg_retriever.retrieve(
            sample['target'],
            series_data,
            sample['forecast_date']
        )
        
        # Prepare context for model
        context_data = self.kg_retriever.prepare_context_for_model(kg_context)
        
        return {
            'patch_embeddings': patch_embeddings[sample['target']]['embeddings'],
            'attention_mask': patch_embeddings[sample['target']]['attention_mask'],
            'kg_context': context_data,
            'target_value': sample['target_value'],
            'target': sample['target'],
            'lag': sample['lag'],
            'forecast_date': sample['forecast_date']
        }

class HybridTrainer:
    """
    Trainer for the hybrid time series forecasting system.
    
    This trainer orchestrates:
    1. Knowledge graph construction
    2. Model training with KG context
    3. Evaluation and logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self._setup_components()
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        
    def _setup_components(self):
        """Setup knowledge graph and patch embedding components"""
        
        # Knowledge graph configuration
        kg_patch_config = KGPatchConfig(
            window_sizes=self.config['kg']['window_sizes'],
            strides=self.config['kg']['strides']
        )
        
        # Build knowledge graph
        self.kg_builder = KnowledgeGraphBuilder(
            db_path=self.config['kg']['db_path'],
            patch_config=kg_patch_config
        )
        
        # GraphRAG retriever
        retrieval_config = RetrievalConfig(
            max_nodes=self.config['retrieval']['max_nodes'],
            max_edges=self.config['retrieval']['max_edges'],
            similarity_threshold=self.config['retrieval']['similarity_threshold']
        )
        
        self.kg_retriever = GraphRAGRetriever(
            db_path=self.config['kg']['db_path'],
            config=retrieval_config
        )
        
        # Patch embedding pipeline
        patch_config = PatchConfig(
            window_sizes=self.config['patches']['window_sizes'],
            strides=self.config['patches']['strides'],
            embedding_dim=self.config['model']['d_model']
        )
        
        self.patch_pipeline = PatchEmbeddingPipeline(
            patch_config=patch_config,
            embedder_config=self.config['patches']['embedder_config']
        )
        
    def _create_model(self) -> TimeLlaMAWithPEFT:
        """Create the Time-LlaMA model"""
        model_config = TimeLlaMAConfig(
            d_model=self.config['model']['d_model'],
            n_heads=self.config['model']['n_heads'],
            n_layers=self.config['model']['n_layers'],
            d_ff=self.config['model']['d_ff'],
            d_context=self.config['model']['d_context'],
            n_context_heads=self.config['model']['n_context_heads'],
            n_targets=self.config['model']['n_targets'],
            forecast_horizons=self.config['model']['forecast_horizons'],
            use_peft=self.config['model']['use_peft'],
            lora_r=self.config['model']['lora_r'],
            lora_alpha=self.config['model']['lora_alpha']
        )
        
        model = TimeLlaMAWithPEFT(model_config)
        model = model.to(self.device)
        
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        if self.config['model']['use_peft']:
            # Only optimize PEFT parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            # Optimize all parameters
            trainable_params = self.model.parameters()
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        return optimizer
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['max_epochs'],
            eta_min=self.config['training']['min_learning_rate']
        )
        
        return scheduler
    
    def build_knowledge_graph(self, train_data: pd.DataFrame, target_pairs: pd.DataFrame):
        """Build the knowledge graph from training data"""
        logger.info("Building knowledge graph...")
        
        # Create database directory if it doesn't exist
        Path(self.config['kg']['db_path']).parent.mkdir(parents=True, exist_ok=True)
        
        # Build knowledge graph
        self.kg_builder.build_from_dataframe(train_data, target_pairs)
        
        logger.info("Knowledge graph construction complete")
    
    def prepare_data(self, train_data: pd.DataFrame, target_pairs: pd.DataFrame) -> DataLoader:
        """Prepare training data"""
        logger.info("Preparing training data...")
        
        # Fit patch pipeline
        self.patch_pipeline.fit(train_data)
        
        # Create dataset
        dataset = CommodityDataset(
            data=train_data,
            target_pairs=target_pairs,
            patch_pipeline=self.patch_pipeline,
            kg_retriever=self.kg_retriever,
            config=self.config['data']
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Created dataloader with {len(dataset)} samples")
        return dataloader
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            patch_embeddings = batch['patch_embeddings'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target_values = batch['target_value'].to(self.device)
            lags = batch['lag']
            
            # Prepare KG context
            kg_contexts = []
            kg_masks = []
            
            for i in range(len(batch['kg_context'])):
                context = batch['kg_context'][i]
                
                # Convert context to tensors
                if context['nodes']:
                    context_tokens = torch.tensor([
                        node['embedding'] for node in context['nodes']
                    ]).unsqueeze(0).to(self.device)
                    context_mask = torch.ones(1, len(context['nodes'])).to(self.device)
                else:
                    # Empty context
                    context_tokens = torch.zeros(1, 1, self.config['model']['d_context']).to(self.device)
                    context_mask = torch.zeros(1, 1).to(self.device)
                
                kg_contexts.append(context_tokens)
                kg_masks.append(context_mask)
            
            # Pad contexts to same length
            max_context_len = max(ctx.size(1) for ctx in kg_contexts)
            padded_contexts = []
            padded_masks = []
            
            for ctx, mask in zip(kg_contexts, kg_masks):
                if ctx.size(1) < max_context_len:
                    pad_size = max_context_len - ctx.size(1)
                    ctx_padded = torch.cat([
                        ctx,
                        torch.zeros(1, pad_size, ctx.size(2)).to(self.device)
                    ], dim=1)
                    mask_padded = torch.cat([
                        mask,
                        torch.zeros(1, pad_size).to(self.device)
                    ], dim=1)
                else:
                    ctx_padded = ctx
                    mask_padded = mask
                
                padded_contexts.append(ctx_padded)
                padded_masks.append(mask_padded)
            
            # Stack contexts
            kg_context = torch.cat(padded_contexts, dim=0)
            kg_mask = torch.cat(padded_masks, dim=0)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            forecasts = self.model(
                patch_embeddings,
                kg_context,
                kg_mask,
                attention_mask
            )
            
            # Compute loss for each horizon
            loss = 0.0
            for lag in lags.unique():
                horizon_key = f'horizon_{lag.item()}'
                if horizon_key in forecasts:
                    # Get targets for this lag
                    lag_mask = (lags == lag)
                    lag_targets = target_values[lag_mask]
                    lag_forecasts = forecasts[horizon_key][lag_mask]
                    
                    if len(lag_targets) > 0:
                        loss += self.criterion(lag_forecasts.squeeze(), lag_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Similar to training but without gradient updates
                patch_embeddings = batch['patch_embeddings'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_values = batch['target_value'].to(self.device)
                lags = batch['lag']
                
                # Prepare KG context (same as training)
                kg_contexts = []
                kg_masks = []
                
                for i in range(len(batch['kg_context'])):
                    context = batch['kg_context'][i]
                    
                    if context['nodes']:
                        context_tokens = torch.tensor([
                            node['embedding'] for node in context['nodes']
                        ]).unsqueeze(0).to(self.device)
                        context_mask = torch.ones(1, len(context['nodes'])).to(self.device)
                    else:
                        context_tokens = torch.zeros(1, 1, self.config['model']['d_context']).to(self.device)
                        context_mask = torch.zeros(1, 1).to(self.device)
                    
                    kg_contexts.append(context_tokens)
                    kg_masks.append(context_mask)
                
                # Pad and stack contexts
                max_context_len = max(ctx.size(1) for ctx in kg_contexts)
                padded_contexts = []
                padded_masks = []
                
                for ctx, mask in zip(kg_contexts, kg_masks):
                    if ctx.size(1) < max_context_len:
                        pad_size = max_context_len - ctx.size(1)
                        ctx_padded = torch.cat([
                            ctx,
                            torch.zeros(1, pad_size, ctx.size(2)).to(self.device)
                        ], dim=1)
                        mask_padded = torch.cat([
                            mask,
                            torch.zeros(1, pad_size).to(self.device)
                        ], dim=1)
                    else:
                        ctx_padded = ctx
                        mask_padded = mask
                    
                    padded_contexts.append(ctx_padded)
                    padded_masks.append(mask_padded)
                
                kg_context = torch.cat(padded_contexts, dim=0)
                kg_mask = torch.cat(padded_masks, dim=0)
                
                # Forward pass
                forecasts = self.model(
                    patch_embeddings,
                    kg_context,
                    kg_mask,
                    attention_mask
                )
                
                # Compute loss
                loss = 0.0
                for lag in lags.unique():
                    horizon_key = f'horizon_{lag.item()}'
                    if horizon_key in forecasts:
                        lag_mask = (lags == lag)
                        lag_targets = target_values[lag_mask]
                        lag_forecasts = forecasts[horizon_key][lag_mask]
                        
                        if len(lag_targets) > 0:
                            loss += self.criterion(lag_forecasts.squeeze(), lag_targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self, train_data: pd.DataFrame, target_pairs: pd.DataFrame, 
              val_data: pd.DataFrame = None):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Build knowledge graph
        self.build_knowledge_graph(train_data, target_pairs)
        
        # Prepare data
        train_loader = self.prepare_data(train_data, target_pairs)
        val_loader = None
        
        if val_data is not None:
            val_loader = self.prepare_data(val_data, target_pairs)
        
        # Initialize wandb if configured
        if self.config['training']['use_wandb']:
            wandb.init(
                project=self.config['training']['wandb_project'],
                config=self.config
            )
        
        # Training loop
        for epoch in range(self.config['training']['max_epochs']):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            logger.info(f"Epoch {epoch}: {metrics}")
            
            if self.config['training']['use_wandb']:
                wandb.log(metrics, step=epoch)
            
            # Save best model
            current_loss = val_metrics.get('val_loss', train_metrics['train_loss'])
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_model(self.config['training']['save_path'])
                logger.info(f"Saved best model with loss {current_loss:.4f}")
        
        logger.info("Training complete!")
    
    def save_model(self, path: str):
        """Save the model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.config
        }, path)
        
        # Save PEFT adapters if using PEFT
        if self.config['model']['use_peft']:
            peft_path = path.replace('.pth', '_peft')
            self.model.save_peft_adapters(peft_path)
    
    def load_model(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Loaded model from epoch {self.epoch} with loss {self.best_loss:.4f}")

def main():
    """Example training script"""
    
    # Load data
    train_data = pd.read_csv('/data/kaggle_projects/commodity_prediction/kaggle_data/train.csv')
    target_pairs = pd.read_csv('/data/kaggle_projects/commodity_prediction/kaggle_data/target_pairs.csv')
    
    # Configuration
    config = {
        'kg': {
            'db_path': '/data/kaggle_projects/commodity_prediction/database/commodity_kg.db',
            'window_sizes': [7, 14, 28],
            'strides': [1, 3, 7]
        },
        'retrieval': {
            'max_nodes': 128,
            'max_edges': 256,
            'similarity_threshold': 0.1
        },
        'patches': {
            'window_sizes': [7, 14, 28],
            'strides': [1, 3, 7],
            'embedder_config': {
                'architecture': 'hybrid',
                'dropout': 0.1
            }
        },
        'model': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 6,
            'd_ff': 512,
            'd_context': 64,
            'n_context_heads': 2,
            'n_targets': 80,
            'forecast_horizons': [1, 2, 3, 4],
            'use_peft': True,
            'lora_r': 16,
            'lora_alpha': 32
        },
        'data': {
            'min_lookback': 100,
            'min_lookback_days': 50
        },
        'training': {
            'batch_size': 8,
            'max_epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'min_learning_rate': 1e-6,
            'max_grad_norm': 1.0,
            'num_workers': 2,
            'use_wandb': False,
            'wandb_project': 'commodity-forecasting',
            'save_path': '/data/kaggle_projects/commodity_prediction/models/best_model.pth'
        }
    }
    
    # Create trainer
    trainer = HybridTrainer(config)
    
    # Train
    trainer.train(train_data, target_pairs)

if __name__ == "__main__":
    main()
