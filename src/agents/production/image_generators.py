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
            # Extreme Memory Stability for T4 (16GB) share with Ollama: 
            # Sequential offload is slower but reduces VRAM usage to ~2GB.
            self.pipe.enable_sequential_cpu_offload()
            
            # NOTE: We keep VAE in float16 (pipeline default) to match UNet latents 
            # and avoid "Half vs Float" type mismatch errors.
            
            # Additional decoder optimizations to prevent hangs and OOM at final step
            self.pipe.enable_vae_tiling()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_attention_slicing()
            
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

    def _get_embeds(self, prompt: str, tokenizer, text_encoder, num_chunks: Optional[int] = None):
        """Helper to get embeds for a single encoder with synchronized chunking support."""
        import re
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        # Skip BOS/EOS
        input_ids = input_ids[1:-1]
        
        chunk_size = 75
        chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]
        
        # Ensure we have the requested number of chunks (even if empty)
        if num_chunks is not None:
            if len(chunks) < num_chunks:
                empty_chunk = torch.tensor([], dtype=torch.long)
                while len(chunks) < num_chunks:
                    chunks.append(empty_chunk)
            elif len(chunks) > num_chunks:
                chunks = chunks[:num_chunks]
        
        all_embeds = []
        all_pooled = []
        
        for chunk in chunks:
            # Re-wrap with BOS/EOS
            wrapped_chunk = torch.cat([
                torch.tensor([tokenizer.bos_token_id]), 
                chunk, 
                torch.tensor([tokenizer.eos_token_id])
            ]).unsqueeze(0).to(self.device)
            
            # Pad to 77
            pad_len = 77 - wrapped_chunk.shape[1]
            if pad_len > 0:
                wrapped_chunk = torch.cat([wrapped_chunk, torch.full((1, pad_len), tokenizer.pad_token_id).to(self.device)], dim=1)
            
            out = text_encoder(wrapped_chunk, output_hidden_states=True)
            all_embeds.append(out.hidden_states[-2])
            all_pooled.append(out[0])
            
        # Concatenate chunks
        final_embeds = torch.cat(all_embeds, dim=1)
        final_pooled = all_pooled[0] 
        
        return final_embeds, final_pooled

    def _encode_prompt(self, prompt: str, negative_prompt: str = ""):
        """
        Encodes a prompt into embeddings that can be used by the pipeline.
        Synchronizes chunk count between positive and negative prompts.
        """
        # 1. Determine how many chunks we need based on the POSITIVE prompt longest tokenization
        p1_ids = self.pipe.tokenizer(prompt).input_ids
        p2_ids = self.pipe.tokenizer_2(prompt).input_ids
        
        # 75 tokens per chunk (excluding BOS/EOS)
        max_p_tokens = max(len(p1_ids) - 2, len(p2_ids) - 2)
        num_chunks = max(1, (max_p_tokens + 74) // 75)
        num_chunks = min(num_chunks, 3) # Max 225 tokens
        
        # 2. Get embeds for both using the SAME num_chunks
        # Encoder 1 (ViT-L/14)
        p1, _ = self._get_embeds(prompt, self.pipe.tokenizer, self.pipe.text_encoder, num_chunks=num_chunks)
        n1, _ = self._get_embeds(negative_prompt, self.pipe.tokenizer, self.pipe.text_encoder, num_chunks=num_chunks)
        
        # Encoder 2 (ViT-bigG/14)
        p2, p_pooled = self._get_embeds(prompt, self.pipe.tokenizer_2, self.pipe.text_encoder_2, num_chunks=num_chunks)
        n2, n_pooled = self._get_embeds(negative_prompt, self.pipe.tokenizer_2, self.pipe.text_encoder_2, num_chunks=num_chunks)
        
        # Concatenate hidden states
        prompt_embeds = torch.cat([p1, p2], dim=-1)
        negative_prompt_embeds = torch.cat([n1, n2], dim=-1)
        
        return prompt_embeds, p_pooled, negative_prompt_embeds, n_pooled

    def generate(self, prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024, seed: Optional[int] = None) -> str:
        logger.info(f"Generating Image with Diffusers: {prompt[:100]}...")
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        try:
            # Handle Long Prompts
            p_embeds, p_pooled, n_embeds, n_pooled = self._encode_prompt(prompt, negative_prompt)

            output = self.pipe(
                prompt_embeds=p_embeds,
                pooled_prompt_embeds=p_pooled,
                negative_prompt_embeds=n_embeds,
                negative_pooled_prompt_embeds=n_pooled,
                width=width,
                height=height,
                generator=generator,
                num_inference_steps=self.inference_steps,
                guidance_scale=0.0 if self.inference_steps < 10 else 7.5
            )
            
            logger.info("⏳ Denoising complete. Finalizing image (VAE decoding)...")
            image = output.images[0]
            
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            import hashlib
            import time
            hash_object = hashlib.md5(prompt.encode())
            filename = f"gen_{hash_object.hexdigest()[:8]}.png"
            filepath = os.path.join(output_dir, filename)
            
            image.save(filepath)
            
            # Memory Safety
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return filepath
            
        except Exception as e:
            logger.error(f"Diffusers generation failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Stop swallowing exceptions - raise so pipeline knows we failed
            raise 

    def generate_batch(self, prompts: List[str], negative_prompt: str = "", **kwargs) -> List[str]:
        """
        Generates multiple images in a single call to maximize throughput.
        Supports long prompts by chunking.
        """
        logger.info(f"Batch generating {len(prompts)} images...")
        try:
            # Encode each prompt in the batch
            all_p_embeds = []
            all_p_pooled = []
            all_n_embeds = []
            all_n_pooled = []
            
            for p in prompts:
                pe, pp, ne, np = self._encode_prompt(p, negative_prompt)
                all_p_embeds.append(pe)
                all_p_pooled.append(pp)
                all_n_embeds.append(ne)
                all_n_pooled.append(np)
                
            p_embeds = torch.cat(all_p_embeds, dim=0)
            p_pooled = torch.cat(all_p_pooled, dim=0)
            n_embeds = torch.cat(all_n_embeds, dim=0)
            n_pooled = torch.cat(all_n_pooled, dim=0)

            output = self.pipe(
                prompt_embeds=p_embeds,
                pooled_prompt_embeds=p_pooled,
                negative_prompt_embeds=n_embeds,
                negative_pooled_prompt_embeds=n_pooled,
                num_inference_steps=self.inference_steps,
                guidance_scale=0.0 if self.inference_steps < 10 else 7.5,
                **kwargs
            )
            
            logger.info("⏳ Batch denoising complete. Finalizing images (VAE decoding)...")
            images = output.images
            
            paths = []
            output_dir = "output"
            import hashlib
            
            for i, image in enumerate(images):
                hash_object = hashlib.md5(prompts[i].encode())
                # Use deterministic naming for batch too
                filename = f"gen_{hash_object.hexdigest()[:8]}.png"
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
            # Stop swallowing exceptions - raise so pipeline knows we failed
            raise
