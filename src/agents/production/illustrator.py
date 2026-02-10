from typing import Any, Dict
from src.core.agent import BaseAgent
from src.core.models import Panel
from src.core.image_interface import ImageGeneratorInterface
from src.agents.visual.consistency_manager import ConsistencyManager

class IllustratorAgent(BaseAgent):
    def __init__(self, agent_name: str, image_generator: ImageGeneratorInterface, consistency_manager: ConsistencyManager, config: Dict[str, Any] = None):
        super().__init__(agent_name, config)
        self.image_generator = image_generator
        self.consistency_manager = consistency_manager

    def process(self, panel: Panel, characters: Any) -> Panel:
        """
        Generates the final image for the panel.
        """
        self.logger.info(f"Illustrating Panel {panel.id}...")
        
        # 1. Get the optimized prompt from Consistency Manager
        final_prompt = self.consistency_manager.process(panel, characters)
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
            # Do not raise, just log error so pipeline continues
            
        return panel
