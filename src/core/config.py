"""
Unified Configuration System for Hybrid Time Series Forecasting

This module provides a centralized configuration system that manages all aspects
of the hybrid forecasting system including model parameters, data processing,
knowledge graph construction, and training/inference settings.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class PatchConfig:
    """Configuration for time series patch generation and embedding"""
    window_sizes: List[int] = field(default_factory=lambda: [7, 14, 28])
    strides: List[int] = field(default_factory=lambda: [1, 3, 7])
    embedding_dim: int = 128
    max_patches_per_series: int = 100
    normalize_patches: bool = True
    min_patch_length: int = 5
    
    # Embedder architecture
    embedder_architecture: str = 'hybrid'  # 'cnn', 'transformer', 'hybrid'
    embedder_dropout: float = 0.1

@dataclass
class KnowledgeGraphConfig:
    """Configuration for knowledge graph construction and storage"""
    db_path: str = 'database/commodity_kg.db'
    window_sizes: List[int] = field(default_factory=lambda: [7, 14, 28])
    strides: List[int] = field(default_factory=lambda: [1, 3, 7])
    
    # Correlation and relationship thresholds
    correlation_threshold: float = 0.1
    p_value_threshold: float = 0.1
    
    # Entity extraction
    extract_entities: bool = True
    include_regime_detection: bool = True

@dataclass
class RetrievalConfig:
    """Configuration for GraphRAG retrieval system"""
    max_nodes: int = 128
    max_edges: int = 256
    similarity_threshold: float = 0.1
    ts_patch_ratio: float = 0.7
    max_hops: int = 2
    recency_decay: float = 90.0
    
    # Scoring weights
    similarity_weight: float = 0.5
    market_match_weight: float = 0.2
    recency_weight: float = 0.1
    edge_strength_weight: float = 0.2

@dataclass
class ModelConfig:
    """Configuration for Time-LlaMA model architecture"""
    # Core architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 512
    dropout: float = 0.1
    
    # Cross-attention to KG context
    d_context: int = 64
    n_context_heads: int = 2
    max_context_tokens: int = 256
    
    # Output configuration
    n_targets: int = 80
    forecast_horizons: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    
    # PEFT/LoRA configuration
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Data configuration
    batch_size: int = 8
    num_workers: int = 2
    min_lookback: int = 100
    min_lookback_days: int = 50
    
    # Training parameters
    max_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    min_learning_rate: float = 1e-6
    max_grad_norm: float = 1.0
    
    # Logging and checkpointing
    use_wandb: bool = False
    wandb_project: str = 'commodity-forecasting'
    save_path: str = 'models/best_model.pth'
    log_interval: int = 100
    
    # Validation
    val_split: float = 0.2
    early_stopping_patience: int = 5

@dataclass
class InferenceConfig:
    """Configuration for inference and submission"""
    model_path: str = 'models/best_model.pth'
    kg_db_path: str = 'database/commodity_kg.db'
    
    # Inference parameters
    batch_size: int = 16
    use_cache: bool = True
    cache_size: int = 1000
    
    # Output configuration
    output_path: str = 'submission.csv'
    confidence_threshold: float = 0.5

@dataclass
class SystemConfig:
    """Main system configuration that combines all components"""
    # Component configurations
    patches: PatchConfig = field(default_factory=PatchConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # System-wide settings
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    seed: int = 42
    debug: bool = False
    log_level: str = 'INFO'
    
    # Data paths
    data_dir: str = 'kaggle_data'
    train_data_path: str = 'kaggle_data/train.csv'
    test_data_path: str = 'kaggle_data/test.csv'
    target_pairs_path: str = 'kaggle_data/target_pairs.csv'

class ConfigManager:
    """
    Configuration manager that handles loading, saving, and validation of configurations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = SystemConfig()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> 'ConfigManager':
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Update configuration
        self._update_config(config_dict)
        
        logger.info(f"Loaded configuration from {config_path}")
        return self
    
    def save_config(self, config_path: str) -> 'ConfigManager':
        """Save configuration to file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self._config_to_dict()
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        logger.info(f"Saved configuration to {config_path}")
        return self
    
    def _update_config(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        # Update each component configuration
        for component_name, component_config in config_dict.items():
            if hasattr(self.config, component_name):
                component = getattr(self.config, component_name)
                
                # Update component attributes
                for key, value in component_config.items():
                    if hasattr(component, key):
                        setattr(component, key, value)
                    else:
                        logger.warning(f"Unknown configuration key: {component_name}.{key}")
            else:
                logger.warning(f"Unknown configuration component: {component_name}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {}
        
        for component_name in ['patches', 'knowledge_graph', 'retrieval', 'model', 'training', 'inference']:
            component = getattr(self.config, component_name)
            config_dict[component_name] = {
                key: value for key, value in component.__dict__.items()
                if not key.startswith('_')
            }
        
        # Add system-wide settings
        config_dict['system'] = {
            'device': self.config.device,
            'seed': self.config.seed,
            'debug': self.config.debug,
            'log_level': self.config.log_level,
            'data_dir': self.config.data_dir,
            'train_data_path': self.config.train_data_path,
            'test_data_path': self.config.test_data_path,
            'target_pairs_path': self.config.target_pairs_path
        }
        
        return config_dict
    
    def validate_config(self) -> bool:
        """Validate configuration for consistency and correctness"""
        errors = []
        
        # Validate model configuration
        if self.config.model.d_model % self.config.model.n_heads != 0:
            errors.append("d_model must be divisible by n_heads")
        
        if self.config.model.d_context > self.config.model.d_model:
            errors.append("d_context should not exceed d_model")
        
        # Validate patch configuration
        if not self.config.patches.window_sizes:
            errors.append("window_sizes cannot be empty")
        
        if any(ws <= 0 for ws in self.config.patches.window_sizes):
            errors.append("window_sizes must be positive")
        
        # Validate training configuration
        if self.config.training.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.config.training.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        # Validate retrieval configuration
        if self.config.retrieval.max_nodes <= 0:
            errors.append("max_nodes must be positive")
        
        if not 0 <= self.config.retrieval.ts_patch_ratio <= 1:
            errors.append("ts_patch_ratio must be between 0 and 1")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_device(self) -> str:
        """Get the appropriate device for computation"""
        if self.config.device == 'auto':
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.config.device
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        import logging
        
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/system.log') if not self.config.debug else logging.NullHandler()
            ]
        )
        
        # Set random seed
        if self.config.seed is not None:
            import random
            import numpy as np
            import torch
            
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.seed)
                torch.cuda.manual_seed_all(self.config.seed)

def create_default_config() -> SystemConfig:
    """Create a default configuration"""
    return SystemConfig()

def load_config_from_file(config_path: str) -> ConfigManager:
    """Load configuration from file"""
    return ConfigManager(config_path)

def save_config_to_file(config: SystemConfig, config_path: str):
    """Save configuration to file"""
    manager = ConfigManager()
    manager.config = config
    manager.save_config(config_path)

# Predefined configurations for different scenarios
def get_kaggle_config() -> SystemConfig:
    """Get configuration optimized for Kaggle environment"""
    config = SystemConfig()
    
    # Optimize for Kaggle constraints
    config.model.d_model = 128
    config.model.n_layers = 4
    config.training.batch_size = 4
    config.training.max_epochs = 10
    config.retrieval.max_nodes = 64
    config.retrieval.max_edges = 128
    
    return config

def get_development_config() -> SystemConfig:
    """Get configuration for development and testing"""
    config = SystemConfig()
    
    # Smaller model for faster development
    config.model.d_model = 64
    config.model.n_layers = 2
    config.training.batch_size = 2
    config.training.max_epochs = 2
    config.patches.window_sizes = [7, 14]
    config.patches.max_patches_per_series = 20
    config.debug = True
    config.log_level = 'DEBUG'
    
    return config

def get_production_config() -> SystemConfig:
    """Get configuration for production deployment"""
    config = SystemConfig()
    
    # Larger model for better performance
    config.model.d_model = 256
    config.model.n_layers = 8
    config.training.batch_size = 16
    config.training.max_epochs = 50
    config.retrieval.max_nodes = 256
    config.retrieval.max_edges = 512
    
    return config
