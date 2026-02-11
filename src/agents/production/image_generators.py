import os
import logging
from typing import Optional, List
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

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(p, **kwargs) for p in prompts]

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
            # Fix upcast_vae deprecation: explicitly move VAE to float32
            self.pipe.vae.to(dtype=torch.float32)
            
            # Enable memory optimizations for Colab T4
            self.pipe.enable_model_cpu_offload() 
            
            # Speed Optimization: Use faster scheduler for Turbo/Lightning if detected
            if "turbo" in model_id.lower() or "lightning" in model_id.lower():
                 from diffusers import EulerDiscreteScheduler
                 self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
                 self.inference_steps = 4 # Turbo needs very few steps
            else:
                 self.inference_steps = 25 # Standard high quality
                 
            logger.info(f"Model loaded successfully. Using {self.inference_steps} steps.")
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
                num_inference_steps=self.inference_steps,
                guidance_scale=0.0 if self.inference_steps < 10 else 7.5 # Lower guidance for Turbo/Lightning
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
            
            # Memory Safety: Clear cache after each generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return filepath
            
        except Exception as e:
            logger.error(f"Diffusers generation failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def generate_batch(self, prompts: List[str], negative_prompt: str = "", **kwargs) -> List[str]:
        """
        Generates multiple images in a single call to maximize throughput.
        """
        logger.info(f"Batch generating {len(prompts)} images...")
        try:
            images = self.pipe(
                prompt=prompts,
                negative_prompt=[negative_prompt] * len(prompts),
                num_inference_steps=self.inference_steps,
                guidance_scale=0.0 if self.inference_steps < 10 else 7.5,
                **kwargs
            ).images
            
            paths = []
            output_dir = "output"
            import hashlib
            import time
            
            for i, image in enumerate(images):
                hash_object = hashlib.md5(prompts[i].encode())
                filename = f"gen_batch_{i}_{hash_object.hexdigest()[:8]}_{int(time.time())}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
                paths.append(filepath)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return paths
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [self.generate(p, negative_prompt=negative_prompt, **kwargs) for p in prompts]
