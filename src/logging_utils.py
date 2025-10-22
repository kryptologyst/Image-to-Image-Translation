"""
Logging and error handling utilities for the Image-to-Image Translation project.

This module provides centralized logging configuration and custom exception classes
for better error handling and debugging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import traceback
import functools


class ImageTranslationError(Exception):
    """Base exception for image translation errors."""
    pass


class ModelLoadingError(ImageTranslationError):
    """Raised when model loading fails."""
    pass


class ImageProcessingError(ImageTranslationError):
    """Raised when image processing fails."""
    pass


class ConfigurationError(ImageTranslationError):
    """Raised when configuration is invalid."""
    pass


class ValidationError(ImageTranslationError):
    """Raised when input validation fails."""
    pass


class LoggingManager:
    """Centralized logging management."""
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[Path] = None):
        """
        Initialize logging manager.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
        """
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_file = log_file
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance."""
        return logging.getLogger(name)


def error_handler(func):
    """
    Decorator for error handling and logging.
    
    This decorator catches exceptions, logs them, and optionally re-raises
    or returns a default value.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    This decorator logs the execution time of functions for performance monitoring.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper


def validate_image_input(image_path: Path) -> None:
    """
    Validate image input file.
    
    Args:
        image_path: Path to image file
        
    Raises:
        ValidationError: If image is invalid
    """
    if not image_path.exists():
        raise ValidationError(f"Image file does not exist: {image_path}")
    
    if not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        raise ValidationError(f"Unsupported image format: {image_path.suffix}")
    
    # Try to open the image to check if it's valid
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        raise ValidationError(f"Invalid image file: {e}")


def validate_model_config(config: Dict[str, Any]) -> None:
    """
    Validate model configuration.
    
    Args:
        config: Model configuration dictionary
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    required_keys = ['model_name', 'torch_dtype']
    
    for key in required_keys:
        if key not in config:
            raise ConfigurationError(f"Missing required configuration key: {key}")
    
    valid_dtypes = ['float16', 'float32', 'bfloat16']
    if config['torch_dtype'] not in valid_dtypes:
        raise ConfigurationError(f"Invalid torch_dtype: {config['torch_dtype']}. Must be one of {valid_dtypes}")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_dir: Optional[Path] = None
) -> LoggingManager:
    """
    Setup application logging.
    
    Args:
        log_level: Logging level
        log_file: Specific log file path
        log_dir: Directory for log files
        
    Returns:
        LoggingManager instance
    """
    if log_file is None and log_dir is not None:
        log_file = log_dir / f"image_translation_{datetime.now().strftime('%Y%m%d')}.log"
    
    return LoggingManager(log_level=log_level, log_file=log_file)


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize performance monitor.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.metrics: Dict[str, Any] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.metrics[f"{operation}_start"] = datetime.now()
    
    def end_timer(self, operation: str) -> float:
        """
        End timing an operation and log the duration.
        
        Args:
            operation: Operation name
            
        Returns:
            Duration in seconds
        """
        start_time = self.metrics.get(f"{operation}_start")
        if start_time is None:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"{operation} completed in {duration:.2f} seconds")
        
        # Store metric
        self.metrics[f"{operation}_duration"] = duration
        return duration
    
    def log_memory_usage(self) -> None:
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")


def create_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Create a logger with consistent formatting.
    
    Args:
        name: Logger name
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    return logger


# Global logging setup
def initialize_logging(config: Optional[Dict[str, Any]] = None) -> LoggingManager:
    """
    Initialize global logging configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        LoggingManager instance
    """
    if config is None:
        config = {}
    
    log_level = config.get('log_level', 'INFO')
    log_dir = Path(config.get('log_dir', 'logs'))
    
    return setup_logging(log_level=log_level, log_dir=log_dir)


if __name__ == "__main__":
    # Example usage
    logging_manager = initialize_logging()
    logger = logging_manager.get_logger(__name__)
    
    logger.info("Logging system initialized")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test error handling decorator
    @error_handler
    @log_execution_time
    def test_function():
        """Test function for decorators."""
        import time
        time.sleep(0.1)
        return "Success"
    
    result = test_function()
    logger.info(f"Test function result: {result}")
