"""
Image-to-Image Translation Module

This module provides a modern, type-safe implementation of image-to-image translation
using state-of-the-art diffusion models from Hugging Face.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import warnings

import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline
import matplotlib.pyplot as plt
from transformers import pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageTranslator:
    """
    A modern image-to-image translation class using Hugging Face diffusion models.
    
    This class provides various image translation capabilities including:
    - Sketch to photo conversion
    - Style transfer
    - Image inpainting
    - Text-guided image editing
    """
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16
    ) -> None:
        """
        Initialize the ImageTranslator.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            torch_dtype: PyTorch data type for model weights
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.pipeline = None
        self.inpaint_pipeline = None
        
        logger.info(f"Initializing ImageTranslator on device: {self.device}")
        self._load_models()
    
    def _load_models(self) -> None:
        """Load the required diffusion models."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load main pipeline for text-to-image and image-to-image
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Load inpainting pipeline for mask-based editing
            self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def sketch_to_photo(
        self,
        input_image: Union[str, Path, Image.Image],
        prompt: str = "a realistic photo",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8
    ) -> Image.Image:
        """
        Convert a sketch to a realistic photo.
        
        Args:
            input_image: Input sketch image (path or PIL Image)
            prompt: Text prompt describing the desired output
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            strength: How much to modify the input image (0-1)
            
        Returns:
            Generated photo as PIL Image
        """
        logger.info("Starting sketch-to-photo translation")
        
        # Load and preprocess image
        if isinstance(input_image, (str, Path)):
            image = Image.open(input_image).convert("RGB")
        else:
            image = input_image.convert("RGB")
        
        # Use img2img pipeline for sketch-to-photo
        result = self.pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength
        ).images[0]
        
        logger.info("Sketch-to-photo translation completed")
        return result
    
    def style_transfer(
        self,
        input_image: Union[str, Path, Image.Image],
        style_prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.7
    ) -> Image.Image:
        """
        Apply style transfer to an image.
        
        Args:
            input_image: Input image (path or PIL Image)
            style_prompt: Style description (e.g., "in the style of Van Gogh")
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            strength: How much to modify the input image (0-1)
            
        Returns:
            Style-transferred image as PIL Image
        """
        logger.info(f"Starting style transfer with prompt: {style_prompt}")
        
        # Load and preprocess image
        if isinstance(input_image, (str, Path)):
            image = Image.open(input_image).convert("RGB")
        else:
            image = input_image.convert("RGB")
        
        result = self.pipeline(
            prompt=style_prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength
        ).images[0]
        
        logger.info("Style transfer completed")
        return result
    
    def inpaint_image(
        self,
        input_image: Union[str, Path, Image.Image],
        mask_image: Union[str, Path, Image.Image],
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Inpaint (fill in) masked regions of an image.
        
        Args:
            input_image: Input image (path or PIL Image)
            mask_image: Mask image (white = inpaint, black = keep)
            prompt: Description of what to paint in the masked area
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            
        Returns:
            Inpainted image as PIL Image
        """
        logger.info("Starting image inpainting")
        
        # Load and preprocess images
        if isinstance(input_image, (str, Path)):
            image = Image.open(input_image).convert("RGB")
        else:
            image = input_image.convert("RGB")
            
        if isinstance(mask_image, (str, Path)):
            mask = Image.open(mask_image).convert("RGB")
        else:
            mask = mask_image.convert("RGB")
        
        result = self.inpaint_pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        logger.info("Image inpainting completed")
        return result
    
    def batch_translate(
        self,
        input_images: List[Union[str, Path, Image.Image]],
        prompts: List[str],
        translation_type: str = "sketch_to_photo",
        **kwargs
    ) -> List[Image.Image]:
        """
        Perform batch translation on multiple images.
        
        Args:
            input_images: List of input images
            prompts: List of prompts (one per image)
            translation_type: Type of translation to perform
            **kwargs: Additional arguments for the translation method
            
        Returns:
            List of translated images
        """
        logger.info(f"Starting batch translation of {len(input_images)} images")
        
        results = []
        for i, (image, prompt) in enumerate(zip(input_images, prompts)):
            logger.info(f"Processing image {i+1}/{len(input_images)}")
            
            if translation_type == "sketch_to_photo":
                result = self.sketch_to_photo(image, prompt, **kwargs)
            elif translation_type == "style_transfer":
                result = self.style_transfer(image, prompt, **kwargs)
            else:
                raise ValueError(f"Unknown translation type: {translation_type}")
            
            results.append(result)
        
        logger.info("Batch translation completed")
        return results
    
    def visualize_comparison(
        self,
        original: Image.Image,
        translated: Image.Image,
        title: str = "Image Translation Comparison",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Visualize original and translated images side by side.
        
        Args:
            original: Original image
            translated: Translated image
            title: Plot title
            save_path: Optional path to save the visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        axes[1].imshow(translated)
        axes[1].set_title("Translated")
        axes[1].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "cuda_available": torch.cuda.is_available(),
            "pipeline_loaded": self.pipeline is not None,
            "inpaint_pipeline_loaded": self.inpaint_pipeline is not None
        }


def create_sample_images(output_dir: Union[str, Path]) -> None:
    """
    Create sample images for testing and demonstration.
    
    Args:
        output_dir: Directory to save sample images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating sample images")
    
    # Create a simple sketch-like image
    sketch = Image.new('RGB', (512, 512), color='white')
    sketch_array = np.array(sketch)
    
    # Draw a simple house sketch
    sketch_array[200:300, 150:350] = [0, 0, 0]  # House body
    sketch_array[150:200, 200:300] = [0, 0, 0]  # Roof
    sketch_array[250:300, 200:250] = [0, 0, 0]  # Door
    sketch_array[180:220, 180:200] = [0, 0, 0]  # Window 1
    sketch_array[180:220, 300:320] = [0, 0, 0]  # Window 2
    
    sketch_image = Image.fromarray(sketch_array)
    sketch_path = output_dir / "sample_sketch.jpg"
    sketch_image.save(sketch_path)
    
    # Create a simple mask for inpainting demo
    mask = Image.new('RGB', (512, 512), color='black')
    mask_array = np.array(mask)
    mask_array[200:300, 200:300] = [255, 255, 255]  # White square to inpaint
    mask_image = Image.fromarray(mask_array)
    mask_path = output_dir / "sample_mask.jpg"
    mask_image.save(mask_path)
    
    logger.info(f"Sample images created in: {output_dir}")


if __name__ == "__main__":
    # Example usage
    translator = ImageTranslator()
    
    # Create sample data
    create_sample_images("data/samples")
    
    # Example: Sketch to photo
    sketch_path = "data/samples/sample_sketch.jpg"
    if os.path.exists(sketch_path):
        result = translator.sketch_to_photo(
            sketch_path,
            prompt="a beautiful house in a garden",
            num_inference_steps=20  # Reduced for faster demo
        )
        
        # Visualize result
        original = Image.open(sketch_path)
        translator.visualize_comparison(
            original,
            result,
            title="Sketch to Photo Translation"
        )
