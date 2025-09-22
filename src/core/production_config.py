"""
Production Configuration System

This module provides a unified, production-ready configuration system that
consolidates all system settings into a single, well-organized structure
optimized for Kaggle commodity forecasting.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import logging
# from omegaconf import OmegaConf  # Not used in current implementation

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """System-level configuration"""
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42
    log_level: str = "INFO"
    experiment_name: str = "commodity_forecasting"
    use_wandb: bool = False
    wandb_project: str = "commodity-forecasting"


@dataclass
class DataConfig:
    """Data processing configuration"""
    raw_path: str = "data/raw"
    processed_path: str = "data/processed"
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    target_pairs_file: str = "target_pairs.csv"
    train_labels_file: str = "train_labels.csv"
    
    # Preprocessing
    fill_method: str = "forward_backward"  # forward, backward, forward_backward
    normalize: bool = True
    remove_outliers: bool = False
    outlier_threshold: float = 3.0


@dataclass
class PatchConfig:
    """Patch embedding configuration"""
    window_sizes: List[int] = field(default_factory=lambda: [7, 14])
    strides: List[int] = field(default_factory=lambda: [7])
    embedding_dim: int = 64
    patch_size: int = 7
    max_patches_per_series: int = 100


@dataclass
class KnowledgeGraphConfig:
    """Knowledge graph configuration"""
    db_path: str = "database/commodity_kg.db"
    correlation_threshold: float = 0.3
    p_value_threshold: float = 0.05
    max_correlations_per_node: int = 50
    cache_size: int = 1000
    
    # Retrieval settings
    max_retrieval_nodes: int = 50
    max_retrieval_edges: int = 100
    similarity_threshold: float = 0.2
    max_hops: int = 1


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Architecture
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 256
    
    # Cross-attention
    context_dim: int = 64
    context_heads: int = 4
    
    # PEFT/LoRA
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Output
    num_horizons: int = 4
    output_dim: int = 1


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic settings
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    
    # Optimization
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    
    # Validation
    val_split: float = 0.2
    early_stopping_patience: int = 10
    save_best_only: bool = True
    
    # Loss
    loss_function: str = "mse"
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "forecast": 1.0,
        "embedding": 0.1
    })


@dataclass
class InferenceConfig:
    """Inference configuration"""
    batch_size: int = 64
    use_graphrag: bool = True
    context_length: int = 50
    output_path: str = "outputs/predictions.csv"
    
    # Kaggle specific
    submission_format: str = "kaggle"
    max_inference_time: int = 3600  # 1 hour


@dataclass
class ProductionConfig:
    """Unified production configuration"""
    system: SystemConfig = field(default_factory=SystemConfig)
    data: DataConfig = field(default_factory=DataConfig)
    patches: PatchConfig = field(default_factory=PatchConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Environment
    environment: str = "development"
    version: str = "1.0.0"


class ProductionConfigManager:
    """Production configuration manager with environment support"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, environment: str = "development") -> ProductionConfig:
        """
        Load configuration for a specific environment.
        
        Args:
            environment: Environment name (development, production, kaggle)
            
        Returns:
            ProductionConfig object
        """
        # Load base configuration
        base_config_path = self.config_dir / "unified_config.yaml"
        if base_config_path.exists():
            base_config = self._load_yaml_config(base_config_path)
        else:
            base_config = {}
        
        # Load environment-specific overrides
        env_config_path = self.config_dir / "environments" / f"{environment}.yaml"
        if env_config_path.exists():
            env_config = self._load_yaml_config(env_config_path)
            base_config = self._deep_merge(base_config, env_config)
        else:
            logger.warning(f"Environment config not found: {env_config_path}")
        
        # Create configuration object
        config = ProductionConfig()
        config.environment = environment
        
        # Apply configuration values
        self._apply_config(config, base_config)
        
        logger.info(f"Loaded configuration for environment: {environment}")
        return config
    
    def save_config(self, config: ProductionConfig, environment: str = None):
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            environment: Environment name (optional)
        """
        if environment is None:
            environment = config.environment
        
        # Convert to dictionary
        config_dict = self._config_to_dict(config)
        
        # Save base configuration
        base_config_path = self.config_dir / "unified_config.yaml"
        self._save_yaml_config(config_dict, base_config_path)
        
        # Save environment-specific overrides
        env_config_path = self.config_dir / "environments" / f"{environment}.yaml"
        self._save_yaml_config(config_dict, env_config_path)
        
        logger.info(f"Saved configuration for environment: {environment}")
    
    def validate_config(self, config: ProductionConfig) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Validate system config
        if config.system.device not in ["cpu", "cuda", "mps"]:
            issues.append(f"Invalid device: {config.system.device}")
        
        if config.system.num_workers < 0:
            issues.append(f"Invalid num_workers: {config.system.num_workers}")
        
        # Validate data config
        if not Path(config.data.raw_path).exists():
            issues.append(f"Raw data path does not exist: {config.data.raw_path}")
        
        # Validate model config
        if config.model.d_model % config.model.n_heads != 0:
            issues.append(f"d_model ({config.model.d_model}) must be divisible by n_heads ({config.model.n_heads})")
        
        if config.model.lora_r <= 0:
            issues.append(f"Invalid lora_r: {config.model.lora_r}")
        
        # Validate training config
        if config.training.batch_size <= 0:
            issues.append(f"Invalid batch_size: {config.training.batch_size}")
        
        if config.training.learning_rate <= 0:
            issues.append(f"Invalid learning_rate: {config.training.learning_rate}")
        
        # Validate KG config
        if config.knowledge_graph.correlation_threshold < 0 or config.knowledge_graph.correlation_threshold > 1:
            issues.append(f"Invalid correlation_threshold: {config.knowledge_graph.correlation_threshold}")
        
        return issues
    
    def _load_yaml_config(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return {}
    
    def _save_yaml_config(self, config: Dict[str, Any], path: Path):
        """Save YAML configuration file"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config to {path}: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_config(self, config_obj: ProductionConfig, config_dict: Dict[str, Any]):
        """Apply configuration dictionary to configuration object"""
        for section_name, section_config in config_dict.items():
            if hasattr(config_obj, section_name):
                section_obj = getattr(config_obj, section_name)
                for key, value in section_config.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                    else:
                        logger.warning(f"Unknown config key: {section_name}.{key}")
    
    def _config_to_dict(self, config: ProductionConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        result = {}
        
        for section_name in ['system', 'data', 'patches', 'knowledge_graph', 'model', 'training', 'inference']:
            if hasattr(config, section_name):
                section_obj = getattr(config, section_name)
                result[section_name] = {}
                
                for key, value in section_obj.__dict__.items():
                    if not key.startswith('_'):
                        result[section_name][key] = value
        
        return result


# Convenience functions
def load_production_config(environment: str = "development") -> ProductionConfig:
    """Load production configuration for specified environment"""
    manager = ProductionConfigManager()
    return manager.load_config(environment)


def save_production_config(config: ProductionConfig, environment: str = None):
    """Save production configuration"""
    manager = ProductionConfigManager()
    manager.save_config(config, environment)


def validate_production_config(config: ProductionConfig) -> bool:
    """Validate production configuration"""
    manager = ProductionConfigManager()
    issues = manager.validate_config(config)
    
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("Configuration validation passed")
    return True
