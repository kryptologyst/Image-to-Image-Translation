"""
Test suite for the Image-to-Image Translation project.

This module contains unit tests and integration tests for the core functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
import torch

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from image_translator import ImageTranslator
from config import ConfigManager, ModelConfig, GenerationConfig, AppConfig
from logging_utils import (
    ImageTranslationError, ModelLoadingError, ImageProcessingError,
    validate_image_input, validate_model_config, PerformanceMonitor
)
from data_generator import create_sample_images, create_simple_shapes_dataset


class TestImageTranslator:
    """Test cases for ImageTranslator class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_image(self, temp_dir):
        """Create a sample image for testing."""
        image_path = temp_dir / "sample.jpg"
        # Create a simple test image
        img = Image.new('RGB', (256, 256), color='white')
        img.save(image_path)
        return image_path
    
    @pytest.fixture
    def mock_translator(self):
        """Create a mock translator for testing without loading actual models."""
        with patch('src.image_translator.StableDiffusionPipeline') as mock_pipeline:
            with patch('src.image_translator.StableDiffusionInpaintPipeline') as mock_inpaint:
                # Mock the pipeline instances
                mock_pipeline.from_pretrained.return_value.to.return_value = Mock()
                mock_inpaint.from_pretrained.return_value.to.return_value = Mock()
                
                translator = ImageTranslator(device="cpu")
                translator.pipeline = Mock()
                translator.inpaint_pipeline = Mock()
                
                # Mock the pipeline methods
                translator.pipeline.return_value.images = [Mock()]
                translator.inpaint_pipeline.return_value.images = [Mock()]
                
                yield translator
    
    def test_translator_initialization(self, mock_translator):
        """Test translator initialization."""
        assert mock_translator.device == "cpu"
        assert mock_translator.model_name == "runwayml/stable-diffusion-v1-5"
        assert mock_translator.pipeline is not None
        assert mock_translator.inpaint_pipeline is not None
    
    def test_sketch_to_photo(self, mock_translator, sample_image):
        """Test sketch to photo translation."""
        result = mock_translator.sketch_to_photo(
            sample_image,
            prompt="test prompt",
            num_inference_steps=10
        )
        
        # Verify pipeline was called
        mock_translator.pipeline.assert_called_once()
        assert result is not None
    
    def test_style_transfer(self, mock_translator, sample_image):
        """Test style transfer functionality."""
        result = mock_translator.style_transfer(
            sample_image,
            style_prompt="test style",
            num_inference_steps=10
        )
        
        # Verify pipeline was called
        mock_translator.pipeline.assert_called_once()
        assert result is not None
    
    def test_inpaint_image(self, mock_translator, sample_image, temp_dir):
        """Test image inpainting functionality."""
        # Create a mask image
        mask_path = temp_dir / "mask.jpg"
        mask_img = Image.new('RGB', (256, 256), color='black')
        mask_img.save(mask_path)
        
        result = mock_translator.inpaint_image(
            sample_image,
            mask_path,
            prompt="test inpaint",
            num_inference_steps=10
        )
        
        # Verify inpaint pipeline was called
        mock_translator.inpaint_pipeline.assert_called_once()
        assert result is not None
    
    def test_batch_translate(self, mock_translator, sample_image):
        """Test batch translation functionality."""
        images = [sample_image, sample_image]
        prompts = ["prompt1", "prompt2"]
        
        results = mock_translator.batch_translate(
            images,
            prompts,
            translation_type="sketch_to_photo",
            num_inference_steps=10
        )
        
        assert len(results) == 2
        assert all(result is not None for result in results)
    
    def test_get_model_info(self, mock_translator):
        """Test model info retrieval."""
        info = mock_translator.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "device" in info
        assert "pipeline_loaded" in info


class TestConfigManager:
    """Test cases for configuration management."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_default_config(self, temp_dir):
        """Test default configuration creation."""
        config_path = temp_dir / "config.yaml"
        config_manager = ConfigManager(config_path)
        
        config = config_manager.get_config()
        assert isinstance(config, AppConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.generation, GenerationConfig)
    
    def test_config_save_load(self, temp_dir):
        """Test configuration save and load."""
        config_path = temp_dir / "test_config.yaml"
        
        # Create and save config
        config_manager = ConfigManager(config_path)
        config_manager.save_config()
        
        # Load config
        new_config_manager = ConfigManager(config_path)
        loaded_config = new_config_manager.get_config()
        
        assert loaded_config.model.model_name == config_manager.get_config().model.model_name
    
    def test_config_update(self, temp_dir):
        """Test configuration updates."""
        config_path = temp_dir / "config.yaml"
        config_manager = ConfigManager(config_path)
        
        # Update configuration
        config_manager.update_config(data_dir="custom_data")
        
        config = config_manager.get_config()
        assert config.data_dir == "custom_data"


class TestLoggingUtils:
    """Test cases for logging utilities."""
    
    def test_validate_image_input(self, temp_dir):
        """Test image input validation."""
        # Create a valid image
        image_path = temp_dir / "test.jpg"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(image_path)
        
        # Should not raise exception
        validate_image_input(image_path)
        
        # Test non-existent file
        non_existent = temp_dir / "nonexistent.jpg"
        with pytest.raises(ValidationError):
            validate_image_input(non_existent)
        
        # Test invalid format
        invalid_path = temp_dir / "test.txt"
        invalid_path.touch()
        with pytest.raises(ValidationError):
            validate_image_input(invalid_path)
    
    def test_validate_model_config(self):
        """Test model configuration validation."""
        # Valid config
        valid_config = {
            'model_name': 'test-model',
            'torch_dtype': 'float16'
        }
        validate_model_config(valid_config)  # Should not raise
        
        # Missing required key
        invalid_config = {'model_name': 'test-model'}
        with pytest.raises(ConfigurationError):
            validate_model_config(invalid_config)
        
        # Invalid dtype
        invalid_dtype_config = {
            'model_name': 'test-model',
            'torch_dtype': 'invalid_dtype'
        }
        with pytest.raises(ConfigurationError):
            validate_model_config(invalid_dtype_config)
    
    def test_performance_monitor(self):
        """Test performance monitoring."""
        import logging
        logger = logging.getLogger(__name__)
        monitor = PerformanceMonitor(logger)
        
        # Test timing
        monitor.start_timer("test_operation")
        import time
        time.sleep(0.01)  # Small delay
        duration = monitor.end_timer("test_operation")
        
        assert duration > 0
        assert "test_operation_duration" in monitor.metrics


class TestDataGenerator:
    """Test cases for data generation utilities."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_create_sample_images(self, temp_dir):
        """Test sample image creation."""
        create_sample_images(temp_dir)
        
        # Check if sample images were created
        assert (temp_dir / "sample_sketch.jpg").exists()
        assert (temp_dir / "sample_mask.jpg").exists()
    
    def test_create_simple_shapes_dataset(self, temp_dir):
        """Test simple shapes dataset creation."""
        output_dir = temp_dir / "shapes"
        create_simple_shapes_dataset(output_dir, num_images=5)
        
        # Check if images were created
        assert output_dir.exists()
        image_files = list(output_dir.glob("*.png"))
        assert len(image_files) == 5


class TestIntegration:
    """Integration tests."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.mark.slow
    def test_end_to_end_workflow(self, temp_dir):
        """Test complete workflow from data generation to translation."""
        # Skip if no GPU available and we're not mocking
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for integration test")
        
        # Create sample data
        create_sample_images(temp_dir)
        
        # Initialize translator (this will actually load models)
        translator = ImageTranslator(device="cpu")  # Use CPU for testing
        
        # Test sketch to photo
        sketch_path = temp_dir / "sample_sketch.jpg"
        if sketch_path.exists():
            result = translator.sketch_to_photo(
                sketch_path,
                prompt="a beautiful house",
                num_inference_steps=5  # Reduced for testing
            )
            assert result is not None


# Pytest configuration
@pytest.fixture(scope="session")
def temp_dir():
    """Session-scoped temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# Test markers
pytestmark = [
    pytest.mark.unit,
    pytest.mark.integration
]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
