"""
Advanced image processing and enhancement utilities.

This module provides state-of-the-art techniques for image preprocessing,
post-processing, and quality enhancement.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Union, List, Tuple, Optional, Dict, Any
import cv2
import torch
import torchvision.transforms as transforms
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Advanced image preprocessing utilities."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.transform_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def enhance_sketch(self, image: Image.Image, method: str = "adaptive") -> Image.Image:
        """
        Enhance sketch images for better translation results.
        
        Args:
            image: Input sketch image
            method: Enhancement method ('adaptive', 'edge_detection', 'contrast')
            
        Returns:
            Enhanced sketch image
        """
        if method == "adaptive":
            return self._adaptive_sketch_enhancement(image)
        elif method == "edge_detection":
            return self._edge_detection_enhancement(image)
        elif method == "contrast":
            return self._contrast_enhancement(image)
        else:
            raise ValueError(f"Unknown enhancement method: {method}")
    
    def _adaptive_sketch_enhancement(self, image: Image.Image) -> Image.Image:
        """Apply adaptive sketch enhancement."""
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to PIL Image
        return Image.fromarray(cleaned).convert('RGB')
    
    def _edge_detection_enhancement(self, image: Image.Image) -> Image.Image:
        """Apply edge detection enhancement."""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to make them thicker
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return Image.fromarray(edges).convert('RGB')
    
    def _contrast_enhancement(self, image: Image.Image) -> Image.Image:
        """Apply contrast enhancement."""
        # Convert to LAB color space for better contrast adjustment
        img_array = np.array(image)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced)
    
    def normalize_image(self, image: Image.Image, target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """
        Normalize image size and aspect ratio.
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            
        Returns:
            Normalized image
        """
        # Resize while maintaining aspect ratio
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create new image with target size and paste centered
        new_image = Image.new('RGB', target_size, (255, 255, 255))
        paste_x = (target_size[0] - image.width) // 2
        paste_y = (target_size[1] - image.height) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        return new_image
    
    def create_mask_from_sketch(self, sketch: Image.Image, threshold: int = 128) -> Image.Image:
        """
        Create a mask from sketch for inpainting.
        
        Args:
            sketch: Input sketch image
            threshold: Threshold for mask creation
            
        Returns:
            Mask image (white = inpaint, black = keep)
        """
        # Convert to grayscale
        gray = sketch.convert('L')
        
        # Create mask based on threshold
        mask_array = np.array(gray)
        mask_array = np.where(mask_array < threshold, 255, 0).astype(np.uint8)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel)
        mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_OPEN, kernel)
        
        return Image.fromarray(mask_array).convert('RGB')


class ImagePostprocessor:
    """Advanced image post-processing utilities."""
    
    def __init__(self):
        """Initialize the postprocessor."""
        pass
    
    def enhance_output(self, image: Image.Image, method: str = "auto") -> Image.Image:
        """
        Enhance generated images for better quality.
        
        Args:
            image: Generated image
            method: Enhancement method ('auto', 'sharpen', 'denoise', 'color_correct')
            
        Returns:
            Enhanced image
        """
        if method == "auto":
            return self._auto_enhancement(image)
        elif method == "sharpen":
            return self._sharpen_image(image)
        elif method == "denoise":
            return self._denoise_image(image)
        elif method == "color_correct":
            return self._color_correction(image)
        else:
            raise ValueError(f"Unknown enhancement method: {method}")
    
    def _auto_enhancement(self, image: Image.Image) -> Image.Image:
        """Apply automatic enhancement based on image analysis."""
        # Analyze image characteristics
        img_array = np.array(image)
        
        # Calculate sharpness (Laplacian variance)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Apply enhancements based on analysis
        enhanced = image.copy()
        
        if sharpness < 100:  # Low sharpness
            enhanced = self._sharpen_image(enhanced)
        
        if brightness < 80:  # Low brightness
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.2)
        
        # Always apply slight contrast enhancement
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        return enhanced
    
    def _sharpen_image(self, image: Image.Image) -> Image.Image:
        """Apply sharpening filter."""
        # Convert to numpy
        img_array = np.array(image)
        
        # Apply unsharp mask
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(img_array, -1, kernel)
        
        # Blend with original
        sharpened = cv2.addWeighted(img_array, 0.7, sharpened, 0.3, 0)
        
        return Image.fromarray(sharpened)
    
    def _denoise_image(self, image: Image.Image) -> Image.Image:
        """Apply denoising filter."""
        img_array = np.array(image)
        
        # Apply bilateral filter for edge-preserving denoising
        denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        return Image.fromarray(denoised)
    
    def _color_correction(self, image: Image.Image) -> Image.Image:
        """Apply color correction."""
        # Convert to LAB color space
        img_array = np.array(image)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Apply histogram equalization to L channel
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        
        # Convert back to RGB
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(corrected)
    
    def blend_images(self, 
                    base_image: Image.Image, 
                    overlay_image: Image.Image, 
                    alpha: float = 0.5,
                    blend_mode: str = "normal") -> Image.Image:
        """
        Blend two images together.
        
        Args:
            base_image: Base image
            overlay_image: Overlay image
            alpha: Blending factor (0-1)
            blend_mode: Blending mode ('normal', 'multiply', 'screen', 'overlay')
            
        Returns:
            Blended image
        """
        # Ensure same size
        if base_image.size != overlay_image.size:
            overlay_image = overlay_image.resize(base_image.size, Image.Resampling.LANCZOS)
        
        # Convert to numpy arrays
        base_array = np.array(base_image, dtype=np.float32)
        overlay_array = np.array(overlay_image, dtype=np.float32)
        
        if blend_mode == "normal":
            blended = (1 - alpha) * base_array + alpha * overlay_array
        elif blend_mode == "multiply":
            blended = base_array * overlay_array / 255.0
        elif blend_mode == "screen":
            blended = 255 - (255 - base_array) * (255 - overlay_array) / 255.0
        elif blend_mode == "overlay":
            mask = base_array < 128
            blended = np.where(mask, 
                             2 * base_array * overlay_array / 255.0,
                             255 - 2 * (255 - base_array) * (255 - overlay_array) / 255.0)
        else:
            raise ValueError(f"Unknown blend mode: {blend_mode}")
        
        # Ensure values are in valid range
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return Image.fromarray(blended)


class QualityAssessment:
    """Image quality assessment utilities."""
    
    def __init__(self):
        """Initialize quality assessor."""
        pass
    
    def calculate_metrics(self, original: Image.Image, generated: Image.Image) -> Dict[str, float]:
        """
        Calculate quality metrics between original and generated images.
        
        Args:
            original: Original image
            generated: Generated image
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Ensure same size
        if original.size != generated.size:
            generated = generated.resize(original.size, Image.Resampling.LANCZOS)
        
        # Convert to numpy arrays
        orig_array = np.array(original, dtype=np.float32)
        gen_array = np.array(generated, dtype=np.float32)
        
        # Calculate MSE
        mse = np.mean((orig_array - gen_array) ** 2)
        metrics['mse'] = mse
        
        # Calculate PSNR
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = float('inf')
        metrics['psnr'] = psnr
        
        # Calculate SSIM (simplified version)
        ssim = self._calculate_ssim(orig_array, gen_array)
        metrics['ssim'] = ssim
        
        # Calculate sharpness
        orig_sharpness = self._calculate_sharpness(orig_array)
        gen_sharpness = self._calculate_sharpness(gen_array)
        metrics['sharpness_original'] = orig_sharpness
        metrics['sharpness_generated'] = gen_sharpness
        
        return metrics
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index (simplified)."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Calculate means
        mu1 = np.mean(gray1)
        mu2 = np.mean(gray2)
        
        # Calculate variances and covariance
        sigma1_sq = np.var(gray1)
        sigma2_sq = np.var(gray2)
        sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
        
        # SSIM constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # Calculate SSIM
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()


class AdvancedImageTranslator:
    """Enhanced image translator with advanced preprocessing and postprocessing."""
    
    def __init__(self, base_translator):
        """
        Initialize enhanced translator.
        
        Args:
            base_translator: Base ImageTranslator instance
        """
        self.base_translator = base_translator
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = ImagePostprocessor()
        self.quality_assessor = QualityAssessment()
    
    def enhanced_sketch_to_photo(self, 
                               input_image: Union[str, Path, Image.Image],
                               prompt: str = "a realistic photo",
                               preprocess: bool = True,
                               postprocess: bool = True,
                               **kwargs) -> Tuple[Image.Image, Dict[str, float]]:
        """
        Enhanced sketch to photo translation with preprocessing and postprocessing.
        
        Args:
            input_image: Input sketch image
            prompt: Text prompt
            preprocess: Whether to apply preprocessing
            postprocess: Whether to apply postprocessing
            **kwargs: Additional arguments for base translation
            
        Returns:
            Tuple of (enhanced_image, quality_metrics)
        """
        # Load image
        if isinstance(input_image, (str, Path)):
            image = Image.open(input_image).convert("RGB")
        else:
            image = input_image.convert("RGB")
        
        original_image = image.copy()
        
        # Preprocessing
        if preprocess:
            image = self.preprocessor.enhance_sketch(image, method="adaptive")
            image = self.preprocessor.normalize_image(image)
        
        # Base translation
        result = self.base_translator.sketch_to_photo(
            image, prompt=prompt, **kwargs
        )
        
        # Postprocessing
        if postprocess:
            result = self.postprocessor.enhance_output(result, method="auto")
        
        # Calculate quality metrics
        metrics = self.quality_assessor.calculate_metrics(original_image, result)
        
        return result, metrics
    
    def batch_enhanced_translate(self,
                                input_images: List[Union[str, Path, Image.Image]],
                                prompts: List[str],
                                translation_type: str = "sketch_to_photo",
                                **kwargs) -> List[Tuple[Image.Image, Dict[str, float]]]:
        """
        Enhanced batch translation with quality assessment.
        
        Args:
            input_images: List of input images
            prompts: List of prompts
            translation_type: Type of translation
            **kwargs: Additional arguments
            
        Returns:
            List of tuples (image, metrics)
        """
        results = []
        
        for i, (image, prompt) in enumerate(zip(input_images, prompts)):
            logger.info(f"Processing enhanced batch item {i+1}/{len(input_images)}")
            
            if translation_type == "sketch_to_photo":
                result, metrics = self.enhanced_sketch_to_photo(
                    image, prompt, **kwargs
                )
            else:
                # Fallback to base translator
                result = self.base_translator.sketch_to_photo(image, prompt, **kwargs)
                metrics = {}
            
            results.append((result, metrics))
        
        return results


if __name__ == "__main__":
    # Example usage
    from image_translator import ImageTranslator
    
    # Initialize base translator
    base_translator = ImageTranslator(device="cpu")
    
    # Create enhanced translator
    enhanced_translator = AdvancedImageTranslator(base_translator)
    
    # Example: Enhanced sketch to photo
    # result, metrics = enhanced_translator.enhanced_sketch_to_photo(
    #     "path/to/sketch.jpg",
    #     prompt="a beautiful landscape",
    #     preprocess=True,
    #     postprocess=True
    # )
    
    # print(f"Quality metrics: {metrics}")
