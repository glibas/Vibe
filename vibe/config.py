"""Configuration management for Vibe."""

import yaml
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = 'saliency_guided'  # 'basic', 'saliency_guided', 'roi_enhanced'
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    n_classes: int = 1
    use_beauty_tokens: bool = True
    saliency_weight: float = 0.5
    roi_scales: list = None
    roi_fusion_method: str = 'attention'
    
    def __post_init__(self):
        if self.roi_scales is None:
            self.roi_scales = [32, 64, 128]


@dataclass  
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    eval_interval: int = 5
    save_interval: int = 10
    gradient_clip: float = 1.0
    
    # Loss weights
    beauty_weight: float = 1.0
    aspect_weight: float = 0.5
    principle_weight: float = 0.3
    consistency_weight: float = 0.2


@dataclass
class DataConfig:
    """Data configuration."""
    train_data_path: str = ""
    train_annotations: str = ""
    val_data_path: str = ""
    val_annotations: str = ""
    test_data_path: str = ""
    test_annotations: str = ""
    
    img_size: int = 224
    num_workers: int = 4
    pin_memory: bool = True
    augmentation: bool = True
    
    # Dataset splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class ExperimentConfig:
    """Overall experiment configuration."""
    name: str = "vibe_experiment"
    description: str = ""
    seed: int = 42
    device: str = "cuda"
    
    # Directories
    output_dir: str = "./outputs"
    log_dir: str = "./logs" 
    checkpoint_dir: str = "./checkpoints"
    
    # Logging
    log_level: str = "INFO"
    log_interval: int = 100
    
    # Evaluation
    eval_metrics: list = None
    
    def __post_init__(self):
        if self.eval_metrics is None:
            self.eval_metrics = ['mse', 'mae', 'rmse', 'pearson', 'spearman']


class ConfigManager:
    """Configuration manager for loading and saving configs."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config = json.load(f)
            else:
                raise ValueError("Config file must be YAML or JSON")
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            save_path: Path to save configuration
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif save_path.endswith('.json'):
                json.dump(config, f, indent=2)
            else:
                raise ValueError("Save path must be YAML or JSON")
    
    @staticmethod
    def create_configs_from_dict(config_dict: Dict[str, Any]) -> tuple:
        """
        Create typed config objects from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Tuple of (model_config, training_config, data_config, experiment_config)
        """
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        experiment_config = ExperimentConfig(**config_dict.get('experiment', {}))
        
        return model_config, training_config, data_config, experiment_config
    
    @staticmethod
    def configs_to_dict(
        model_config: ModelConfig,
        training_config: TrainingConfig, 
        data_config: DataConfig,
        experiment_config: ExperimentConfig
    ) -> Dict[str, Any]:
        """Convert config objects to dictionary."""
        return {
            'model': asdict(model_config),
            'training': asdict(training_config),
            'data': asdict(data_config),
            'experiment': asdict(experiment_config),
        }


def load_config(config_path: str):
    """Convenience function to load configuration."""
    return ConfigManager.load_config(config_path)


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    experiment_config = ExperimentConfig()
    
    return ConfigManager.configs_to_dict(
        model_config, training_config, data_config, experiment_config
    )


def save_default_config(save_path: str = "./config/default.yaml"):
    """Save default configuration to file."""
    config = create_default_config()
    ConfigManager.save_config(config, save_path)