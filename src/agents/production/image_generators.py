import os
import logging
from typing import Optional
from src.core.image_interface import ImageGeneratorInterface

logger = logging.getLogger(__name__)

# Lazy import checks
try:
    import torch
    from diffusers import DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    torch = None # Placeholder

class MockImageGenerator(ImageGeneratorInterface):
    """
    Mock generator for testing the pipeline without GPU.
    """
    def generate(self, prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024, seed: Optional[int] = None) -> str:
        logger.info(f"Mock Generating Image with prompt: {prompt[:50]}...")
        # Create a dummy image file
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate a unique filename based on prompt hash or timestamp
        import hashlib
        import time
        hash_object = hashlib.md5(prompt.encode())
        filename = f"mock_{hash_object.hexdigest()[:8]}_{int(time.time())}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Create a simple colored placeholder image using PIL
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (width, height), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((10,10), "Mock Image", fill=(255,255,0))
        d.text((10,30), prompt[:100], fill=(255,255,255))
        img.save(filepath)
        
        return filepath

class DiffusersImageGenerator(ImageGeneratorInterface):
    """
    Real generator using Hugging Face Diffusers (Stable Diffusion XL / Flux).
    Requires GPU.
    """
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0", device: str = "cuda"):
        self.device = device
        self.model_id = model_id
        
        logger.info(f"Loading Diffusers model: {model_id} on {device}...")
        try:
            from diffusers import DiffusionPipeline
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                use_safetensors=True, 
                variant="fp16"
            )
            self.pipe.to(device)
            # Enable memory optimizations for Colab T4
            self.pipe.enable_model_cpu_offload() 
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Diffusers model: {e}")
            raise

    def generate(self, prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024, seed: Optional[int] = None) -> str:
        logger.info(f"Generating Image with Diffusers: {prompt[:50]}...")
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        try:
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                generator=generator,
                num_inference_steps=30 # Good balance for speed/quality
            ).images[0]
            
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            import hashlib
            import time
            hash_object = hashlib.md5(prompt.encode())
            filename = f"gen_{hash_object.hexdigest()[:8]}_{int(time.time())}.png"
            filepath = os.path.join(output_dir, filename)
            
            image.save(filepath)
            return filepath
            
        except Exception as e:
            logger.error(f"Diffusers generation failed: {e}")
            raise
