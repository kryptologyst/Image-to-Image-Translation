"""
Data generation utilities for creating synthetic datasets and sample images.

This module provides functions to create mock data for testing and demonstration
purposes when real datasets are not available.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import random

logger = logging.getLogger(__name__)


def create_simple_shapes_dataset(
    output_dir: Path,
    num_images: int = 50,
    image_size: Tuple[int, int] = (512, 512)
) -> None:
    """
    Create a dataset of simple geometric shapes for testing.
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to generate
        image_size: Size of generated images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating {num_images} simple shape images")
    
    for i in range(num_images):
        # Create white background
        img = Image.new('RGB', image_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Randomly choose shape type
        shape_type = random.choice(['circle', 'rectangle', 'triangle', 'line'])
        
        # Generate random colors and positions
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        x1 = random.randint(50, image_size[0] - 100)
        y1 = random.randint(50, image_size[1] - 100)
        x2 = random.randint(x1 + 50, image_size[0] - 50)
        y2 = random.randint(y1 + 50, image_size[1] - 50)
        
        if shape_type == 'circle':
            draw.ellipse([x1, y1, x2, y2], fill=color, outline='black', width=3)
        elif shape_type == 'rectangle':
            draw.rectangle([x1, y1, x2, y2], fill=color, outline='black', width=3)
        elif shape_type == 'triangle':
            points = [(x1, y2), (x2, y2), ((x1 + x2) // 2, y1)]
            draw.polygon(points, fill=color, outline='black', width=3)
        elif shape_type == 'line':
            draw.line([x1, y1, x2, y2], fill='black', width=5)
        
        # Save image
        img.save(output_dir / f"shape_{i:03d}.png")


def create_sketch_dataset(
    output_dir: Path,
    num_images: int = 30,
    image_size: Tuple[int, int] = (512, 512)
) -> None:
    """
    Create a dataset of sketch-like images for sketch-to-photo translation.
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to generate
        image_size: Size of generated images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating {num_images} sketch images")
    
    sketch_templates = [
        "house", "tree", "car", "face", "flower", "mountain", "sun", "cloud"
    ]
    
    for i in range(num_images):
        # Create white background
        img = Image.new('RGB', image_size, color='white')
        draw = ImageDraw.Draw(img)
        
        template = random.choice(sketch_templates)
        
        if template == "house":
            _draw_house_sketch(draw, image_size)
        elif template == "tree":
            _draw_tree_sketch(draw, image_size)
        elif template == "car":
            _draw_car_sketch(draw, image_size)
        elif template == "face":
            _draw_face_sketch(draw, image_size)
        elif template == "flower":
            _draw_flower_sketch(draw, image_size)
        elif template == "mountain":
            _draw_mountain_sketch(draw, image_size)
        elif template == "sun":
            _draw_sun_sketch(draw, image_size)
        elif template == "cloud":
            _draw_cloud_sketch(draw, image_size)
        
        # Save image
        img.save(output_dir / f"sketch_{i:03d}.png")


def _draw_house_sketch(draw: ImageDraw.Draw, size: Tuple[int, int]) -> None:
    """Draw a simple house sketch."""
    w, h = size
    # House body
    draw.rectangle([w//3, h//2, 2*w//3, 4*h//5], outline='black', width=3)
    # Roof
    draw.polygon([(w//3, h//2), (2*w//3, h//2), (w//2, h//3)], outline='black', width=3)
    # Door
    draw.rectangle([w//2-20, 3*h//5, w//2+20, 4*h//5], outline='black', width=3)
    # Windows
    draw.rectangle([w//3+20, h//2+20, w//2-20, 3*h//5-20], outline='black', width=3)
    draw.rectangle([w//2+20, h//2+20, 2*w//3-20, 3*h//5-20], outline='black', width=3)


def _draw_tree_sketch(draw: ImageDraw.Draw, size: Tuple[int, int]) -> None:
    """Draw a simple tree sketch."""
    w, h = size
    # Trunk
    draw.rectangle([w//2-10, 3*h//5, w//2+10, 4*h//5], outline='black', width=3)
    # Leaves (circle)
    draw.ellipse([w//3, h//5, 2*w//3, 3*h//5], outline='black', width=3)


def _draw_car_sketch(draw: ImageDraw.Draw, size: Tuple[int, int]) -> None:
    """Draw a simple car sketch."""
    w, h = size
    # Car body
    draw.rectangle([w//4, 2*h//5, 3*w//4, 3*h//5], outline='black', width=3)
    # Wheels
    draw.ellipse([w//4-15, 3*h//5-15, w//4+15, 3*h//5+15], outline='black', width=3)
    draw.ellipse([3*w//4-15, 3*h//5-15, 3*w//4+15, 3*h//5+15], outline='black', width=3)


def _draw_face_sketch(draw: ImageDraw.Draw, size: Tuple[int, int]) -> None:
    """Draw a simple face sketch."""
    w, h = size
    # Face outline
    draw.ellipse([w//3, h//4, 2*w//3, 3*h//4], outline='black', width=3)
    # Eyes
    draw.ellipse([w//2-30, h//2-20, w//2-10, h//2], outline='black', width=2)
    draw.ellipse([w//2+10, h//2-20, w//2+30, h//2], outline='black', width=2)
    # Nose
    draw.polygon([(w//2, h//2), (w//2-10, h//2+20), (w//2+10, h//2+20)], outline='black', width=2)
    # Mouth
    draw.arc([w//2-20, h//2+30, w//2+20, h//2+50], 0, 180, fill='black', width=2)


def _draw_flower_sketch(draw: ImageDraw.Draw, size: Tuple[int, int]) -> None:
    """Draw a simple flower sketch."""
    w, h = size
    # Stem
    draw.line([w//2, h//2, w//2, 4*h//5], fill='black', width=3)
    # Flower petals
    for i in range(6):
        angle = i * 60
        x = int(w//2 + 30 * np.cos(np.radians(angle)))
        y = int(h//2 + 30 * np.sin(np.radians(angle)))
        draw.ellipse([x-10, y-10, x+10, y+10], outline='black', width=2)
    # Center
    draw.ellipse([w//2-8, h//2-8, w//2+8, h//2+8], outline='black', width=2)


def _draw_mountain_sketch(draw: ImageDraw.Draw, size: Tuple[int, int]) -> None:
    """Draw a simple mountain sketch."""
    w, h = size
    # Mountain peaks
    draw.polygon([(w//4, 3*h//4), (w//2, h//3), (3*w//4, 3*h//4)], outline='black', width=3)
    # Base line
    draw.line([w//8, 3*h//4, 7*w//8, 3*h//4], fill='black', width=3)


def _draw_sun_sketch(draw: ImageDraw.Draw, size: Tuple[int, int]) -> None:
    """Draw a simple sun sketch."""
    w, h = size
    # Sun circle
    draw.ellipse([w//2-40, h//4-40, w//2+40, h//4+40], outline='black', width=3)
    # Sun rays
    for i in range(8):
        angle = i * 45
        x1 = int(w//2 + 50 * np.cos(np.radians(angle)))
        y1 = int(h//4 + 50 * np.sin(np.radians(angle)))
        x2 = int(w//2 + 70 * np.cos(np.radians(angle)))
        y2 = int(h//4 + 70 * np.sin(np.radians(angle)))
        draw.line([x1, y1, x2, y2], fill='black', width=3)


def _draw_cloud_sketch(draw: ImageDraw.Draw, size: Tuple[int, int]) -> None:
    """Draw a simple cloud sketch."""
    w, h = size
    # Cloud shape (multiple circles)
    draw.ellipse([w//3, h//3, w//3+60, h//3+40], outline='black', width=3)
    draw.ellipse([w//3+30, h//3-10, w//3+90, h//3+30], outline='black', width=3)
    draw.ellipse([w//3+60, h//3, w//3+120, h//3+40], outline='black', width=3)


def create_mask_dataset(
    output_dir: Path,
    num_images: int = 20,
    image_size: Tuple[int, int] = (512, 512)
) -> None:
    """
    Create a dataset of mask images for inpainting.
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to generate
        image_size: Size of generated images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating {num_images} mask images")
    
    for i in range(num_images):
        # Create black background
        img = Image.new('RGB', image_size, color='black')
        draw = ImageDraw.Draw(img)
        
        # Create random white regions (areas to inpaint)
        mask_type = random.choice(['rectangle', 'circle', 'polygon', 'multiple'])
        
        if mask_type == 'rectangle':
            x1 = random.randint(50, image_size[0] - 150)
            y1 = random.randint(50, image_size[1] - 150)
            x2 = random.randint(x1 + 50, image_size[0] - 50)
            y2 = random.randint(y1 + 50, image_size[1] - 50)
            draw.rectangle([x1, y1, x2, y2], fill='white')
            
        elif mask_type == 'circle':
            x = random.randint(100, image_size[0] - 100)
            y = random.randint(100, image_size[1] - 100)
            radius = random.randint(30, 80)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill='white')
            
        elif mask_type == 'polygon':
            points = []
            center_x, center_y = image_size[0]//2, image_size[1]//2
            for _ in range(random.randint(3, 6)):
                angle = random.uniform(0, 2 * np.pi)
                radius = random.randint(30, 80)
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                points.append((x, y))
            draw.polygon(points, fill='white')
            
        elif mask_type == 'multiple':
            # Multiple small regions
            for _ in range(random.randint(2, 5)):
                x = random.randint(50, image_size[0] - 50)
                y = random.randint(50, image_size[1] - 50)
                size = random.randint(20, 60)
                draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], fill='white')
        
        # Save image
        img.save(output_dir / f"mask_{i:03d}.png")


def create_sample_data(data_dir: Path) -> None:
    """
    Create all sample datasets for testing and demonstration.
    
    Args:
        data_dir: Root data directory
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating sample datasets")
    
    # Create different types of sample data
    create_simple_shapes_dataset(data_dir / "shapes", num_images=20)
    create_sketch_dataset(data_dir / "sketches", num_images=15)
    create_mask_dataset(data_dir / "masks", num_images=10)
    
    logger.info("Sample datasets created successfully")


if __name__ == "__main__":
    # Create sample data
    data_dir = Path("data")
    create_sample_data(data_dir)
