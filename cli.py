#!/usr/bin/env python3
"""
Command Line Interface for Image-to-Image Translation.

This module provides a CLI for the image translation functionality,
making it easy to use from the command line.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from image_translator import ImageTranslator
from advanced_processing import AdvancedImageTranslator
from config import get_config, get_model_config
from logging_utils import setup_logging, validate_image_input
from data_generator import create_sample_data

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Image-to-Image Translation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert sketch to photo
  python cli.py sketch-to-photo input.jpg --prompt "a beautiful house"
  
  # Apply style transfer
  python cli.py style-transfer input.jpg --style "in the style of Van Gogh"
  
  # Inpaint image
  python cli.py inpaint input.jpg mask.jpg --prompt "a garden"
  
  # Batch processing
  python cli.py batch sketches/ --prompt "realistic photo" --output outputs/
  
  # Create sample data
  python cli.py create-samples --output data/samples
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for results"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Sketch to photo command
    sketch_parser = subparsers.add_parser(
        "sketch-to-photo",
        help="Convert sketch to photo"
    )
    sketch_parser.add_argument("input", type=Path, help="Input sketch image")
    sketch_parser.add_argument("--prompt", "-p", default="a realistic photo", help="Text prompt")
    sketch_parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    sketch_parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    sketch_parser.add_argument("--strength", type=float, default=0.8, help="Strength")
    sketch_parser.add_argument("--enhanced", action="store_true", help="Use enhanced processing")
    
    # Style transfer command
    style_parser = subparsers.add_parser(
        "style-transfer",
        help="Apply style transfer"
    )
    style_parser.add_argument("input", type=Path, help="Input image")
    style_parser.add_argument("--style", "-s", required=True, help="Style prompt")
    style_parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    style_parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    style_parser.add_argument("--strength", type=float, default=0.7, help="Strength")
    
    # Inpaint command
    inpaint_parser = subparsers.add_parser(
        "inpaint",
        help="Inpaint masked regions"
    )
    inpaint_parser.add_argument("input", type=Path, help="Input image")
    inpaint_parser.add_argument("mask", type=Path, help="Mask image")
    inpaint_parser.add_argument("--prompt", "-p", required=True, help="Inpainting prompt")
    inpaint_parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    inpaint_parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    
    # Batch processing command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch process multiple images"
    )
    batch_parser.add_argument("input_dir", type=Path, help="Input directory")
    batch_parser.add_argument("--prompt", "-p", required=True, help="Prompt for all images")
    batch_parser.add_argument("--type", choices=["sketch_to_photo", "style_transfer"], 
                            default="sketch_to_photo", help="Translation type")
    batch_parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    batch_parser.add_argument("--enhanced", action="store_true", help="Use enhanced processing")
    
    # Create samples command
    samples_parser = subparsers.add_parser(
        "create-samples",
        help="Create sample images for testing"
    )
    samples_parser.add_argument("--output", "-o", type=Path, default=Path("data/samples"),
                               help="Output directory")
    samples_parser.add_argument("--count", type=int, default=10, help="Number of samples")
    
    # Web app command
    web_parser = subparsers.add_parser(
        "web",
        help="Launch web interface"
    )
    web_parser.add_argument("--port", type=int, default=8501, help="Port number")
    web_parser.add_argument("--host", default="localhost", help="Host address")
    
    return parser


def setup_logging_from_args(args) -> None:
    """Setup logging based on command line arguments."""
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = None
    
    if args.output:
        log_file = args.output / "translation.log"
    
    setup_logging(log_level=log_level, log_file=log_file)


def handle_sketch_to_photo(args) -> None:
    """Handle sketch to photo command."""
    logger.info(f"Converting sketch to photo: {args.input}")
    
    # Validate input
    validate_image_input(args.input)
    
    # Initialize translator
    device = args.device if args.device != "auto" else None
    translator = ImageTranslator(device=device)
    
    if args.enhanced:
        enhanced_translator = AdvancedImageTranslator(translator)
        result, metrics = enhanced_translator.enhanced_sketch_to_photo(
            args.input,
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            strength=args.strength
        )
        logger.info(f"Quality metrics: {metrics}")
    else:
        result = translator.sketch_to_photo(
            args.input,
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            strength=args.strength
        )
    
    # Save result
    output_path = args.output / f"sketch_to_photo_{args.input.stem}.jpg" if args.output else f"sketch_to_photo_{args.input.stem}.jpg"
    result.save(output_path)
    logger.info(f"Result saved to: {output_path}")


def handle_style_transfer(args) -> None:
    """Handle style transfer command."""
    logger.info(f"Applying style transfer: {args.input}")
    
    # Validate input
    validate_image_input(args.input)
    
    # Initialize translator
    device = args.device if args.device != "auto" else None
    translator = ImageTranslator(device=device)
    
    result = translator.style_transfer(
        args.input,
        style_prompt=args.style,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        strength=args.strength
    )
    
    # Save result
    output_path = args.output / f"style_transfer_{args.input.stem}.jpg" if args.output else f"style_transfer_{args.input.stem}.jpg"
    result.save(output_path)
    logger.info(f"Result saved to: {output_path}")


def handle_inpaint(args) -> None:
    """Handle inpainting command."""
    logger.info(f"Inpainting image: {args.input}")
    
    # Validate inputs
    validate_image_input(args.input)
    validate_image_input(args.mask)
    
    # Initialize translator
    device = args.device if args.device != "auto" else None
    translator = ImageTranslator(device=device)
    
    result = translator.inpaint_image(
        args.input,
        args.mask,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance
    )
    
    # Save result
    output_path = args.output / f"inpaint_{args.input.stem}.jpg" if args.output else f"inpaint_{args.input.stem}.jpg"
    result.save(output_path)
    logger.info(f"Result saved to: {output_path}")


def handle_batch(args) -> None:
    """Handle batch processing command."""
    logger.info(f"Batch processing images in: {args.input_dir}")
    
    # Get input images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    input_images = [f for f in args.input_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not input_images:
        logger.error(f"No images found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(input_images)} images to process")
    
    # Initialize translator
    device = args.device if args.device != "auto" else None
    translator = ImageTranslator(device=device)
    
    # Create output directory
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    
    # Process images
    prompts = [args.prompt] * len(input_images)
    
    if args.enhanced:
        enhanced_translator = AdvancedImageTranslator(translator)
        results = enhanced_translator.batch_enhanced_translate(
            input_images,
            prompts,
            translation_type=args.type,
            num_inference_steps=args.steps
        )
        
        for i, (result, metrics) in enumerate(results):
            output_path = args.output / f"batch_{i:03d}.jpg" if args.output else f"batch_{i:03d}.jpg"
            result.save(output_path)
            logger.info(f"Batch item {i+1} saved to: {output_path}")
            logger.info(f"Quality metrics: {metrics}")
    else:
        results = translator.batch_translate(
            input_images,
            prompts,
            translation_type=args.type,
            num_inference_steps=args.steps
        )
        
        for i, result in enumerate(results):
            output_path = args.output / f"batch_{i:03d}.jpg" if args.output else f"batch_{i:03d}.jpg"
            result.save(output_path)
            logger.info(f"Batch item {i+1} saved to: {output_path}")


def handle_create_samples(args) -> None:
    """Handle create samples command."""
    logger.info(f"Creating sample images in: {args.output}")
    
    args.output.mkdir(parents=True, exist_ok=True)
    create_sample_data(args.output)
    
    logger.info(f"Created sample images in: {args.output}")


def handle_web(args) -> None:
    """Handle web app command."""
    logger.info(f"Launching web interface on {args.host}:{args.port}")
    
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "web_app/app.py",
        "--server.port", str(args.port),
        "--server.address", args.host
    ]
    
    subprocess.run(cmd)


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging_from_args(args)
    
    # Create output directory if specified
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    
    try:
        # Handle commands
        if args.command == "sketch-to-photo":
            handle_sketch_to_photo(args)
        elif args.command == "style-transfer":
            handle_style_transfer(args)
        elif args.command == "inpaint":
            handle_inpaint(args)
        elif args.command == "batch":
            handle_batch(args)
        elif args.command == "create-samples":
            handle_create_samples(args)
        elif args.command == "web":
            handle_web(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        logger.info("Command completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
