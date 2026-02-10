from src.core.agent import BaseAgent
from src.core.models import Panel
from src.core.image_interface import ImageGeneratorInterface
from src.agents.visual.consistency_manager import ConsistencyManager
from typing import Any, Dict, List, Optional

class IllustratorAgent(BaseAgent):
    def __init__(self, agent_name: str, image_generator: ImageGeneratorInterface, consistency_manager: ConsistencyManager, config: Dict[str, Any] = None):
        super().__init__(agent_name, config)
        self.image_generator = image_generator
        self.consistency_manager = consistency_manager

    def process(self, panel: Panel, characters: Any, style_guide: Optional[str] = None) -> Panel:
        """
        Generates the final image for the panel (sequential fallback).
        """
        self.logger.info(f"Illustrating Panel {panel.id}...")
        
        # 1. Get the optimized prompt from Consistency Manager
        final_prompt = self.consistency_manager.process(panel, characters, style_guide=style_guide)
        panel.image_prompt = final_prompt
        
        # 2. Call Image Generator
        try:
            image_path = self.image_generator.generate(
                prompt=final_prompt,
                negative_prompt=self.config.get("negative_prompt", "blurry, weird text, bad anatomy"),
                width=self.config.get("width", 1024),
                height=self.config.get("height", 1024)
            )
            panel.image_path = image_path
            self.logger.info(f"Generated image saved at: {image_path}")
        except Exception as e:
            self.logger.error(f"Image generation failed for Panel {panel.id}: {e}")
            
        return panel

    def run_batch(self, panels: List[Panel], characters: Any, style_guide: Optional[str] = None) -> List[Panel]:
        """
        Generates images for multiple panels in parallel/batch.
        """
        self.logger.info(f"🎨 Batch Illustrating {len(panels)} panels...")
        
        # 1. Prepare Prompts
        prompts = []
        for panel in panels:
            final_prompt = self.consistency_manager.process(panel, characters, style_guide=style_guide)
            panel.image_prompt = final_prompt
            prompts.append(final_prompt)
            
        # 2. Call Batch Image Generator
        try:
            image_paths = self.image_generator.generate_batch(
                prompts=prompts,
                negative_prompt=self.config.get("negative_prompt", "blurry, weird text, bad anatomy")
            )
            
            for i, path in enumerate(image_paths):
                if i < len(panels):
                    panels[i].image_path = path
                    self.logger.info(f"Generated image for Panel {panels[i].id} saved at: {path}")
        except Exception as e:
            self.logger.error(f"Batch image generation failed: {e}. Falling back to sequential.")
            # Fallback to sequential
            for panel in panels:
                self.process(panel, characters)
                
        return panels
