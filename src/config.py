"""
Configuration management for the Image-to-Image Translation project.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    model_name: str = "runwayml/stable-diffusion-v1-5"
    device: Optional[str] = None
    torch_dtype: str = "float16"
    safety_checker: bool = False


@dataclass
class GenerationConfig:
    """Configuration for image generation parameters."""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    strength: float = 0.8
    width: int = 512
    height: int = 512


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig
    generation: GenerationConfig
    data_dir: str = "data"
    output_dir: str = "outputs"
    log_level: str = "INFO"
    max_batch_size: int = 4


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or Path("config/config.yaml")
        self.config: Optional[AppConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            logger.info(f"Loading configuration from: {self.config_path}")
            self._load_from_file()
        else:
            logger.info("No config file found, using default configuration")
            self._create_default_config()
    
    def _load_from_file(self) -> None:
        """Load configuration from YAML or JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                    data = yaml.safe_load(f)
                elif self.config_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")
            
            # Convert dict to AppConfig
            self.config = AppConfig(
                model=ModelConfig(**data.get('model', {})),
                generation=GenerationConfig(**data.get('generation', {})),
                data_dir=data.get('data_dir', 'data'),
                output_dir=data.get('output_dir', 'outputs'),
                log_level=data.get('log_level', 'INFO'),
                max_batch_size=data.get('max_batch_size', 4)
            )
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create default configuration."""
        self.config = AppConfig(
            model=ModelConfig(),
            generation=GenerationConfig()
        )
        
        # Save default config
        self.save_config()
    
    def save_config(self, path: Optional[Path] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Optional path to save config (uses default if None)
        """
        save_path = path or self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = asdict(self.config)
            
            with open(save_path, 'w') as f:
                if save_path.suffix.lower() == '.yaml' or save_path.suffix.lower() == '.yml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif save_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {save_path.suffix}")
            
            logger.info(f"Configuration saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration values.
        
        Args:
            **kwargs: Configuration values to update
        """
        if self.config is None:
            self._create_default_config()
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.config.model
    
    def get_generation_config(self) -> GenerationConfig:
        """Get generation configuration."""
        return self.config.generation


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config_manager.get_config()


def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return config_manager.get_model_config()


def get_generation_config() -> GenerationConfig:
    """Get generation configuration."""
    return config_manager.get_generation_config()
