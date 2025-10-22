"""
Streamlit web interface for Image-to-Image Translation.

This module provides a user-friendly web interface for the image translation
capabilities using Streamlit.
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Optional, List
import logging
from PIL import Image
import io
import base64

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from image_translator import ImageTranslator, create_sample_images
from config import get_config, get_model_config, get_generation_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Image-to-Image Translation",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_translator():
    """Load the image translator model (cached for performance)."""
    try:
        config = get_model_config()
        translator = ImageTranslator(
            model_name=config.model_name,
            device=config.device,
            torch_dtype=getattr(__import__('torch'), config.torch_dtype)
        )
        return translator
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for display."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def display_image_comparison(original: Image.Image, translated: Image.Image, title: str = "Comparison"):
    """Display original and translated images side by side."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original, caption="Original", use_column_width=True)
    
    with col2:
        st.image(translated, caption="Translated", use_column_width=True)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎨 Image-to-Image Translation</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Load model
    with st.spinner("Loading AI model..."):
        translator = load_translator()
    
    if translator is None:
        st.error("Failed to load the AI model. Please check your configuration.")
        return
    
    # Model info
    with st.sidebar.expander("Model Information"):
        model_info = translator.get_model_info()
        for key, value in model_info.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["Sketch to Photo", "Style Transfer", "Image Inpainting", "Batch Processing"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Sketch to Photo Translation</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input")
            uploaded_file = st.file_uploader(
                "Upload a sketch image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a sketch or line drawing to convert to a realistic photo"
            )
            
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file).convert("RGB")
                st.image(input_image, caption="Input Sketch", use_column_width=True)
        
        with col2:
            st.subheader("Parameters")
            prompt = st.text_input(
                "Prompt",
                value="a realistic photo",
                help="Describe what you want the output to look like"
            )
            
            num_steps = st.slider("Inference Steps", 10, 100, 50)
            guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
            strength = st.slider("Strength", 0.1, 1.0, 0.8)
            
            if st.button("Generate Photo", type="primary"):
                if uploaded_file is not None:
                    with st.spinner("Generating photo..."):
                        try:
                            result = translator.sketch_to_photo(
                                input_image,
                                prompt=prompt,
                                num_inference_steps=num_steps,
                                guidance_scale=guidance_scale,
                                strength=strength
                            )
                            
                            st.success("Photo generated successfully!")
                            display_image_comparison(input_image, result, "Sketch to Photo")
                            
                        except Exception as e:
                            st.error(f"Error generating photo: {e}")
                else:
                    st.warning("Please upload an image first.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Style Transfer</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input")
            uploaded_file = st.file_uploader(
                "Upload an image for style transfer",
                type=['png', 'jpg', 'jpeg'],
                key="style_input"
            )
            
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file).convert("RGB")
                st.image(input_image, caption="Input Image", use_column_width=True)
        
        with col2:
            st.subheader("Style Parameters")
            style_prompt = st.text_input(
                "Style Prompt",
                value="in the style of Van Gogh",
                help="Describe the artistic style you want to apply"
            )
            
            num_steps = st.slider("Inference Steps", 10, 100, 50, key="style_steps")
            guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, key="style_guidance")
            strength = st.slider("Strength", 0.1, 1.0, 0.7, key="style_strength")
            
            if st.button("Apply Style", type="primary"):
                if uploaded_file is not None:
                    with st.spinner("Applying style..."):
                        try:
                            result = translator.style_transfer(
                                input_image,
                                style_prompt=style_prompt,
                                num_inference_steps=num_steps,
                                guidance_scale=guidance_scale,
                                strength=strength
                            )
                            
                            st.success("Style transfer completed!")
                            display_image_comparison(input_image, result, "Style Transfer")
                            
                        except Exception as e:
                            st.error(f"Error applying style: {e}")
                else:
                    st.warning("Please upload an image first.")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Image Inpainting</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Images")
            uploaded_image = st.file_uploader(
                "Upload an image to inpaint",
                type=['png', 'jpg', 'jpeg'],
                key="inpaint_image"
            )
            
            uploaded_mask = st.file_uploader(
                "Upload a mask image",
                type=['png', 'jpg', 'jpeg'],
                key="inpaint_mask",
                help="White areas will be inpainted, black areas will be preserved"
            )
            
            if uploaded_image is not None and uploaded_mask is not None:
                input_image = Image.open(uploaded_image).convert("RGB")
                mask_image = Image.open(uploaded_mask).convert("RGB")
                
                col_img, col_mask = st.columns(2)
                with col_img:
                    st.image(input_image, caption="Input Image", use_column_width=True)
                with col_mask:
                    st.image(mask_image, caption="Mask", use_column_width=True)
        
        with col2:
            st.subheader("Inpainting Parameters")
            inpaint_prompt = st.text_input(
                "Inpainting Prompt",
                value="a beautiful landscape",
                help="Describe what you want to paint in the masked area"
            )
            
            num_steps = st.slider("Inference Steps", 10, 100, 50, key="inpaint_steps")
            guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, key="inpaint_guidance")
            
            if st.button("Inpaint Image", type="primary"):
                if uploaded_image is not None and uploaded_mask is not None:
                    with st.spinner("Inpainting image..."):
                        try:
                            result = translator.inpaint_image(
                                input_image,
                                mask_image,
                                prompt=inpaint_prompt,
                                num_inference_steps=num_steps,
                                guidance_scale=guidance_scale
                            )
                            
                            st.success("Inpainting completed!")
                            display_image_comparison(input_image, result, "Image Inpainting")
                            
                        except Exception as e:
                            st.error(f"Error inpainting image: {e}")
                else:
                    st.warning("Please upload both image and mask files.")
    
    with tab4:
        st.markdown('<h2 class="sub-header">Batch Processing</h2>', unsafe_allow_html=True)
        
        st.info("Upload multiple images for batch processing. Each image will be processed with the same parameters.")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="batch_files"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")
            
            batch_prompt = st.text_input(
                "Batch Prompt",
                value="a realistic photo",
                help="This prompt will be applied to all images"
            )
            
            translation_type = st.selectbox(
                "Translation Type",
                ["sketch_to_photo", "style_transfer"],
                help="Choose the type of translation to apply"
            )
            
            if st.button("Process Batch", type="primary"):
                if uploaded_files:
                    with st.spinner(f"Processing {len(uploaded_files)} images..."):
                        try:
                            input_images = [Image.open(f).convert("RGB") for f in uploaded_files]
                            prompts = [batch_prompt] * len(input_images)
                            
                            results = translator.batch_translate(
                                input_images,
                                prompts,
                                translation_type=translation_type,
                                num_inference_steps=30  # Reduced for batch processing
                            )
                            
                            st.success(f"Batch processing completed! Processed {len(results)} images.")
                            
                            # Display results in a grid
                            for i, (original, result) in enumerate(zip(input_images, results)):
                                st.subheader(f"Image {i+1}")
                                display_image_comparison(original, result, f"Batch Result {i+1}")
                                
                        except Exception as e:
                            st.error(f"Error in batch processing: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <h4>About this Application</h4>
        <p>This application uses state-of-the-art diffusion models for image-to-image translation. 
        It supports sketch-to-photo conversion, style transfer, image inpainting, and batch processing.</p>
        <p><strong>Note:</strong> Processing time depends on your hardware. GPU acceleration is recommended for best performance.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
