"""
Model Factory for Hybrid Time Series Forecasting System

This module provides a unified interface for creating and managing all model components
including the Time-LlaMA adapter, patch embedders, and PEFT configurations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path

from .config import SystemConfig, ModelConfig, PatchConfig
from ..models.time_llama import SimpleTimeLlaMAWithPEFT, create_simple_time_llama
from ..data.patch_embedder import PatchEmbeddingPipeline, PatchConfig as EmbedderPatchConfig

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class for creating and managing model components.
    
    This class provides a unified interface for creating all model components
    and ensures consistency across the system.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.device = self._get_device()
        
    def _get_device(self) -> torch.device:
        """Get the appropriate device for computation"""
        if self.config.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.config.device)
    
    def create_patch_embedder(self) -> PatchEmbeddingPipeline:
        """
        Create the patch embedding pipeline.
        
        Returns:
            Configured patch embedding pipeline
        """
        logger.info("Creating patch embedder...")
        
        # Convert system patch config to embedder patch config
        patch_config = EmbedderPatchConfig(
            window_sizes=self.config.patches.window_sizes,
            strides=self.config.patches.strides,
            embedding_dim=self.config.patches.embedding_dim,
            max_patches_per_series=self.config.patches.max_patches_per_series,
            normalize_patches=self.config.patches.normalize_patches
        )
        
        # Create embedder configuration
        embedder_config = {
            'architecture': self.config.patches.embedder_architecture,
            'dropout': self.config.patches.embedder_dropout
        }
        
        # Create pipeline
        pipeline = PatchEmbeddingPipeline(
            patch_config=patch_config,
            embedder_config=embedder_config
        )
        
        logger.info(f"Created patch embedder with {patch_config.embedding_dim}D embeddings")
        return pipeline
    
    def create_time_llama_model(self) -> SimpleTimeLlaMAWithPEFT:
        """
        Create the Time-LlaMA model with PEFT.
        
        Returns:
            Configured Time-LlaMA model
        """
        logger.info("Creating Time-LlaMA model...")
        
        # Create model using the simple Time-LlaMA implementation
        model = create_simple_time_llama(
            patch_size=self.config.patches.window_sizes[0],  # Use first window size
            patch_stride=self.config.patches.strides[0],     # Use first stride
            patch_embed_dim=self.config.model.d_model,
            n_layer=self.config.model.n_layers,
            n_head=self.config.model.n_heads,
            n_embd_per_head=self.config.model.d_model // self.config.model.n_heads,
            dropout=self.config.model.dropout,
            context_dim=self.config.model.d_context,
            context_heads=self.config.model.n_context_heads,
            max_context_tokens=self.config.model.max_context_tokens,
            prediction_length=len(self.config.model.forecast_horizons),
            use_peft=self.config.model.use_peft,
        )
        model = model.to(self.device)
        
        # Log model information
        total_params = model.get_num_parameters()
        trainable_params = model.get_trainable_parameters()
        
        logger.info(f"Created Time-LlaMA model:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Trainable ratio: {trainable_params/total_params:.2%}")
        logger.info(f"  Device: {self.device}")
        
        return model
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Create optimizer for the model.
        
        Args:
            model: The model to optimize
            
        Returns:
            Configured optimizer
        """
        logger.info("Creating optimizer...")
        
        # Get trainable parameters
        if self.config.model.use_peft:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
        else:
            trainable_params = model.parameters()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        logger.info(f"Created AdamW optimizer with lr={self.config.training.learning_rate}")
        return optimizer
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            
        Returns:
            Configured scheduler
        """
        logger.info("Creating learning rate scheduler...")
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.training.max_epochs,
            eta_min=self.config.training.min_learning_rate
        )
        
        logger.info(f"Created CosineAnnealingLR scheduler")
        return scheduler
    
    def create_loss_function(self) -> nn.Module:
        """
        Create loss function for training.
        
        Returns:
            Configured loss function
        """
        logger.info("Creating loss function...")
        
        # Use MSE loss for regression
        criterion = nn.MSELoss()
        
        logger.info("Created MSE loss function")
        return criterion
    
    def load_model(self, model_path: str) -> SimpleTimeLlaMAWithPEFT:
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the model checkpoint
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {model_path}")
        
        # Create model
        model = self.create_time_llama_model()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("Model loaded successfully")
        return model
    
    def save_model(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler,
                   epoch: int, loss: float, save_path: str):
        """
        Save model checkpoint.
        
        Args:
            model: The model to save
            optimizer: The optimizer state
            scheduler: The scheduler state
            epoch: Current epoch
            loss: Current loss
            save_path: Path to save the checkpoint
        """
        logger.info(f"Saving model to {save_path}")
        
        # Create save directory
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'config': self.config
        }
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
        
        # Save PEFT adapters if using PEFT
        if self.config.model.use_peft and hasattr(model, 'save_peft_adapters'):
            peft_path = save_path.replace('.pth', '_peft')
            model.save_peft_adapters(peft_path)
        
        logger.info("Model saved successfully")
    
    def get_model_summary(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get a summary of the model.
        
        Args:
            model: The model to summarize
            
        Returns:
            Model summary dictionary
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_ratio': trainable_params / total_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'device': str(self.device),
            'config': {
                'd_model': self.config.model.d_model,
                'n_heads': self.config.model.n_heads,
                'n_layers': self.config.model.n_layers,
                'use_peft': self.config.model.use_peft,
                'lora_r': self.config.model.lora_r if self.config.model.use_peft else None
            }
        }
        
        return summary

class ModelManager:
    """
    Manager class for handling model lifecycle and operations.
    
    This class provides high-level operations for model management including
    training, evaluation, and inference.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.factory = ModelFactory(config)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
    def initialize_model(self) -> 'ModelManager':
        """
        Initialize all model components.
        
        Returns:
            Self for method chaining
        """
        logger.info("Initializing model components...")
        
        # Create model
        self.model = self.factory.create_time_llama_model()
        
        # Create optimizer
        self.optimizer = self.factory.create_optimizer(self.model)
        
        # Create scheduler
        self.scheduler = self.factory.create_scheduler(self.optimizer)
        
        # Create loss function
        self.criterion = self.factory.create_loss_function()
        
        logger.info("Model components initialized")
        return self
    
    def load_checkpoint(self, checkpoint_path: str) -> 'ModelManager':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.factory.device)
        
        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()
        
        # Load state dicts
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info("Checkpoint loaded successfully")
        return self
    
    def save_checkpoint(self, epoch: int, loss: float, save_path: str) -> 'ModelManager':
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            loss: Current loss
            save_path: Path to save checkpoint
            
        Returns:
            Self for method chaining
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.factory.save_model(
            self.model, self.optimizer, self.scheduler, epoch, loss, save_path
        )
        
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Model information dictionary
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        return self.factory.get_model_summary(self.model)
    
    def set_training_mode(self) -> 'ModelManager':
        """Set model to training mode"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.train()
        return self
    
    def set_eval_mode(self) -> 'ModelManager':
        """Set model to evaluation mode"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.eval()
        return self

def create_model_manager(config: SystemConfig) -> ModelManager:
    """
    Create a model manager with the given configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured model manager
    """
    return ModelManager(config)

def create_patch_embedder(config: SystemConfig) -> PatchEmbeddingPipeline:
    """
    Create a patch embedder with the given configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured patch embedder
    """
    factory = ModelFactory(config)
    return factory.create_patch_embedder()
