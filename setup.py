#!/usr/bin/env python3
"""
Setup script for the Image-to-Image Translation project.

This script handles the initial setup, dependency installation,
and configuration of the project.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    logger.info(f"Running: {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")


def install_dependencies():
    """Install project dependencies."""
    logger.info("Installing dependencies...")
    
    try:
        run_command("pip install -r requirements.txt")
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        sys.exit(1)


def create_directories():
    """Create necessary directories."""
    logger.info("Creating project directories...")
    
    directories = [
        "data/samples",
        "data/sketches", 
        "data/masks",
        "outputs",
        "logs",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def create_sample_data():
    """Create sample data for testing."""
    logger.info("Creating sample data...")
    
    try:
        # Import and run data generator
        sys.path.append(str(Path(__file__).parent / "src"))
        from data_generator import create_sample_data
        
        create_sample_data(Path("data"))
        logger.info("Sample data created successfully")
    except Exception as e:
        logger.warning(f"Failed to create sample data: {e}")


def setup_git():
    """Setup git repository if not already initialized."""
    if not Path(".git").exists():
        logger.info("Initializing git repository...")
        run_command("git init")
        run_command("git add .")
        run_command("git commit -m 'Initial commit: Image-to-Image Translation project'")
        logger.info("Git repository initialized")
    else:
        logger.info("Git repository already exists")


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            logger.warning("CUDA is not available. Using CPU for inference.")
    except ImportError:
        logger.warning("PyTorch not installed yet. CUDA check will be done after installation.")


def run_tests():
    """Run project tests."""
    logger.info("Running tests...")
    
    try:
        run_command("python -m pytest tests/ -v")
        logger.info("All tests passed!")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Some tests failed: {e}")
        logger.info("You can run tests manually later with: python -m pytest tests/")


def create_launch_scripts():
    """Create convenient launch scripts."""
    logger.info("Creating launch scripts...")
    
    # Web app launcher
    web_script = """#!/bin/bash
# Launch web interface
streamlit run web_app/app.py --server.port 8501
"""
    Path("launch_web.sh").write_text(web_script)
    Path("launch_web.sh").chmod(0o755)
    
    # CLI launcher
    cli_script = """#!/bin/bash
# Launch CLI interface
python cli.py "$@"
"""
    Path("launch_cli.sh").write_text(cli_script)
    Path("launch_cli.sh").chmod(0o755)
    
    logger.info("Launch scripts created: launch_web.sh, launch_cli.sh")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Image-to-Image Translation project")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-samples", action="store_true", help="Skip creating sample data")
    parser.add_argument("--dev", action="store_true", help="Setup for development")
    
    args = parser.parse_args()
    
    logger.info("Starting Image-to-Image Translation project setup...")
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    if not args.skip_deps:
        install_dependencies()
    
    # Create directories
    create_directories()
    
    # Create sample data
    if not args.skip_samples:
        create_sample_data()
    
    # Check CUDA
    check_cuda()
    
    # Run tests
    if not args.skip_tests:
        run_tests()
    
    # Setup git
    setup_git()
    
    # Create launch scripts
    create_launch_scripts()
    
    logger.info("Setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Launch web interface: ./launch_web.sh")
    logger.info("2. Or use CLI: ./launch_cli.sh --help")
    logger.info("3. Check README.md for detailed usage instructions")
    
    if args.dev:
        logger.info("\nDevelopment setup:")
        logger.info("- Install development dependencies: pip install -e .")
        logger.info("- Run tests: python -m pytest tests/")
        logger.info("- Format code: black src/ tests/")
        logger.info("- Lint code: flake8 src/ tests/")


if __name__ == "__main__":
    main()
