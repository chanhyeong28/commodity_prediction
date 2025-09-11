"""
Unified Model Factory

This module provides a unified interface for creating and managing all model components
in the production-ready commodity forecasting system.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path

from .production_config import ProductionConfig, ProductionConfigManager
from ..models.unified_time_llama import UnifiedTimeLlaMA, TimeLlaMAConfig, create_unified_time_llama
from ..kg.unified_kg_system import UnifiedKGSystem, KGConfig

logger = logging.getLogger(__name__)


class UnifiedModelFactory:
    """
    Unified factory class for creating and managing all system components.
    
    This class provides a single interface for creating models, KG systems,
    and other components in the production system.
    """
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.device = self._get_device()
        self._models = {}
        self._kg_system = None
        
        logger.info(f"Initialized UnifiedModelFactory with device: {self.device}")
    
    def _get_device(self) -> str:
        """Get the appropriate device for computation"""
        if self.config.system.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif self.config.system.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def create_time_llama_model(self) -> UnifiedTimeLlaMA:
        """
        Create a unified Time-LlaMA model.
        
        Returns:
            UnifiedTimeLlaMA model instance
        """
        if 'time_llama' in self._models:
            return self._models['time_llama']
        
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
        model = create_unified_time_llama(model_config, self.device)
        self._models['time_llama'] = model
        
        logger.info(f"Created Time-LlaMA model with {model.get_num_parameters():,} parameters")
        return model
    
    def create_kg_system(self) -> UnifiedKGSystem:
        """
        Create a unified knowledge graph system.
        
        Returns:
            UnifiedKGSystem instance
        """
        if self._kg_system is not None:
            return self._kg_system
        
        # Create KG configuration
        kg_config = KGConfig(
            window_sizes=self.config.patches.window_sizes,
            strides=self.config.patches.strides,
            max_nodes_per_series=self.config.patches.max_patches_per_series,
            embedding_dim=self.config.patches.embedding_dim,
            correlation_threshold=self.config.knowledge_graph.correlation_threshold,
            p_value_threshold=self.config.knowledge_graph.p_value_threshold,
            max_correlations_per_node=self.config.knowledge_graph.max_correlations_per_node,
            max_retrieval_nodes=self.config.knowledge_graph.max_retrieval_nodes,
            max_retrieval_edges=self.config.knowledge_graph.max_retrieval_edges,
            similarity_threshold=self.config.knowledge_graph.similarity_threshold,
            cache_size=self.config.knowledge_graph.cache_size,
            db_path=self.config.knowledge_graph.db_path
        )
        
        # Create KG system
        self._kg_system = UnifiedKGSystem(kg_config)
        
        logger.info(f"Created unified KG system with database: {kg_config.db_path}")
        return self._kg_system
    
    def load_model(self, model_path: str) -> UnifiedTimeLlaMA:
        """
        Load a pre-trained model from file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded UnifiedTimeLlaMA model
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model first
        model = self.create_time_llama_model()
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        logger.info(f"Loaded model from {model_path}")
        return model
    
    def save_model(self, model: UnifiedTimeLlaMA, save_path: str, 
                   additional_info: Optional[Dict[str, Any]] = None):
        """
        Save a model to file.
        
        Args:
            model: Model to save
            save_path: Path to save the model
            additional_info: Additional information to save with the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'device': self.device
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved model to {save_path}")
    
    def get_model_info(self, model: UnifiedTimeLlaMA) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model: Model to get info for
            
        Returns:
            Dictionary containing model information
        """
        return {
            'total_parameters': model.get_num_parameters(),
            'trainable_parameters': model.get_trainable_parameters(),
            'device': self.device,
            'config': self.config
        }
    
    def create_training_components(self) -> Tuple[UnifiedTimeLlaMA, UnifiedKGSystem]:
        """
        Create all components needed for training.
        
        Returns:
            Tuple of (model, kg_system)
        """
        model = self.create_time_llama_model()
        kg_system = self.create_kg_system()
        
        logger.info("Created training components")
        return model, kg_system
    
    def create_inference_components(self) -> Tuple[UnifiedTimeLlaMA, UnifiedKGSystem]:
        """
        Create all components needed for inference.
        
        Returns:
            Tuple of (model, kg_system)
        """
        model = self.create_time_llama_model()
        kg_system = self.create_kg_system()
        
        logger.info("Created inference components")
        return model, kg_system
    
    def cleanup(self):
        """Clean up resources"""
        self._models.clear()
        self._kg_system = None
        logger.info("Cleaned up model factory resources")


def create_unified_factory(environment: str = "development") -> UnifiedModelFactory:
    """
    Create a unified model factory for the specified environment.
    
    Args:
        environment: Environment name (development, production, kaggle)
        
    Returns:
        UnifiedModelFactory instance
    """
    config_manager = ProductionConfigManager()
    config = config_manager.load_config(environment)
    
    return UnifiedModelFactory(config)
