# Image-to-Image Translation

A production-ready implementation of image-to-image translation using state-of-the-art diffusion models from Hugging Face. This project provides various image translation capabilities including sketch-to-photo conversion, style transfer, image inpainting, and batch processing.

## Features

- **Sketch to Photo**: Convert sketches and line drawings into realistic photos
- **Style Transfer**: Apply artistic styles to images
- **Image Inpainting**: Fill in masked regions of images
- **Batch Processing**: Process multiple images simultaneously
- **Web Interface**: User-friendly Streamlit web application
- **Configuration Management**: YAML-based configuration system
- **Logging & Monitoring**: Comprehensive logging and error handling
- **Testing**: Unit tests and validation
- **Documentation**: Comprehensive documentation and examples

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for best performance)
- At least 8GB RAM (16GB recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Image-to-Image-Translation.git
cd Image-to-Image-Translation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create sample data:
```bash
python src/data_generator.py
```

## Quick Start

### Web Interface (Recommended)

Launch the Streamlit web application:

```bash
streamlit run web_app/app.py
```

Open your browser to `http://localhost:8501` and start translating images!

### Command Line Usage

```python
from src.image_translator import ImageTranslator

# Initialize the translator
translator = ImageTranslator()

# Convert sketch to photo
result = translator.sketch_to_photo(
    "path/to/sketch.jpg",
    prompt="a beautiful landscape",
    num_inference_steps=50
)

# Apply style transfer
styled_image = translator.style_transfer(
    "path/to/image.jpg",
    style_prompt="in the style of Van Gogh"
)

# Inpaint masked regions
inpainted = translator.inpaint_image(
    "path/to/image.jpg",
    "path/to/mask.jpg",
    prompt="a beautiful garden"
)
```

## Project Structure

```
image-to-image-translation/
├── src/                    # Source code
│   ├── image_translator.py # Main translation module
│   ├── config.py          # Configuration management
│   └── data_generator.py  # Sample data generation
├── web_app/               # Web interface
│   └── app.py            # Streamlit application
├── data/                  # Data directory
│   ├── samples/          # Sample images
│   ├── sketches/         # Sketch dataset
│   └── masks/            # Mask dataset
├── config/               # Configuration files
│   └── config.yaml       # Default configuration
├── tests/                # Test files
├── models/               # Model cache directory
├── outputs/              # Generated outputs
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Configuration

The application uses YAML configuration files. Create `config/config.yaml`:

```yaml
model:
  model_name: "runwayml/stable-diffusion-v1-5"
  device: null  # Auto-detect
  torch_dtype: "float16"
  safety_checker: false

generation:
  num_inference_steps: 50
  guidance_scale: 7.5
  strength: 0.8
  width: 512
  height: 512

data_dir: "data"
output_dir: "outputs"
log_level: "INFO"
max_batch_size: 4
```

## Usage Examples

### 1. Sketch to Photo Translation

```python
from src.image_translator import ImageTranslator

translator = ImageTranslator()

# Convert a house sketch to a realistic photo
result = translator.sketch_to_photo(
    "data/samples/sketch_001.png",
    prompt="a beautiful house in a garden with flowers",
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=0.8
)

# Visualize the result
translator.visualize_comparison(
    Image.open("data/samples/sketch_001.png"),
    result,
    title="Sketch to Photo Translation"
)
```

### 2. Style Transfer

```python
# Apply Van Gogh style to an image
styled = translator.style_transfer(
    "path/to/image.jpg",
    style_prompt="in the style of Van Gogh, oil painting",
    strength=0.7
)
```

### 3. Batch Processing

```python
# Process multiple images
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
prompts = ["realistic photo", "realistic photo", "realistic photo"]

results = translator.batch_translate(
    images,
    prompts,
    translation_type="sketch_to_photo",
    num_inference_steps=30  # Reduced for batch processing
)
```

### 4. Image Inpainting

```python
# Fill in masked regions
inpainted = translator.inpaint_image(
    "path/to/image.jpg",
    "path/to/mask.jpg",  # White = inpaint, Black = keep
    prompt="a beautiful landscape"
)
```

## API Reference

### ImageTranslator Class

#### `__init__(model_name, device, torch_dtype)`
Initialize the image translator.

**Parameters:**
- `model_name` (str): Hugging Face model identifier
- `device` (str, optional): Device to run inference on
- `torch_dtype` (torch.dtype): PyTorch data type for model weights

#### `sketch_to_photo(input_image, prompt, **kwargs)`
Convert a sketch to a realistic photo.

**Parameters:**
- `input_image`: Input sketch image (path or PIL Image)
- `prompt` (str): Text prompt describing the desired output
- `num_inference_steps` (int): Number of denoising steps
- `guidance_scale` (float): How closely to follow the prompt
- `strength` (float): How much to modify the input image (0-1)

**Returns:** PIL Image

#### `style_transfer(input_image, style_prompt, **kwargs)`
Apply style transfer to an image.

**Parameters:**
- `input_image`: Input image (path or PIL Image)
- `style_prompt` (str): Style description
- `num_inference_steps` (int): Number of denoising steps
- `guidance_scale` (float): How closely to follow the prompt
- `strength` (float): How much to modify the input image (0-1)

**Returns:** PIL Image

#### `inpaint_image(input_image, mask_image, prompt, **kwargs)`
Inpaint masked regions of an image.

**Parameters:**
- `input_image`: Input image (path or PIL Image)
- `mask_image`: Mask image (white = inpaint, black = keep)
- `prompt` (str): Description of what to paint
- `num_inference_steps` (int): Number of denoising steps
- `guidance_scale` (float): How closely to follow the prompt

**Returns:** PIL Image

## Performance Tips

1. **GPU Acceleration**: Use CUDA-compatible GPU for best performance
2. **Batch Processing**: Process multiple images together for efficiency
3. **Model Optimization**: Use `torch.float16` for faster inference
4. **Memory Management**: Adjust batch size based on available memory

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Loading Errors**: Check internet connection and model availability
3. **Slow Performance**: Ensure GPU is being used and drivers are updated

### Performance Optimization

```python
# Use CPU if GPU is not available
translator = ImageTranslator(device="cpu")

# Reduce inference steps for faster generation
result = translator.sketch_to_photo(
    image,
    prompt,
    num_inference_steps=20  # Faster but lower quality
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the excellent diffusion models
- Stability AI for Stable Diffusion
- The open-source community for inspiration and support

## Support

For questions, issues, or contributions, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Join our community discussions


# Image-to-Image-Translation
