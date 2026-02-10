from typing import List, Dict, Any, Optional
from src.core.agent import BaseAgent
from src.core.models import Character, Panel

class ConsistencyManager(BaseAgent):
    def __init__(self, agent_name: str = "ConsistencyManager", config: Dict[str, Any] = None):
        super().__init__(agent_name, config)
        self.style_preset = self.config.get("style_preset", "cinematic, detailed, comic book style")
        self.negative_prompt = self.config.get("negative_prompt", "blurry, low quality, distortion, bad anatomy")

    def process(self, panel: Panel, characters: List[Character], style_guide: Optional[str] = None) -> str:
        """
        Constructs the final image generation prompt for a panel, enforcing character consistency.
        """
        self.logger.info(f"Generating consistent prompt for Panel {panel.id}...")
        
        # Base prompt from panel description
        # Priority: Style Guide from Script > Style Preset from Config
        style = style_guide or self.style_preset
        prompt_parts = [style, panel.description]
        
        # Add character details if present in the panel
        if panel.characters_present:
            for char_name in panel.characters_present:
                # Find the character object (check name or aliases)
                character = None
                for c in characters:
                    if c.name.lower() == char_name.lower() or any(alias.lower() == char_name.lower() for alias in c.aliases):
                        character = c
                        break
                
                if character:
                    # Append strict visual description to the prompt
                    # In a more advanced version, this would trigger IP-Adapter or LoRA loading
                    prompt_parts.append(f"({character.name}: {character.description}, {character.personality} expression)")
        
        # Add camera/lighting if specified
        if panel.camera_angle:
            prompt_parts.append(f"Camera: {panel.camera_angle}")
        if panel.lighting:
            prompt_parts.append(f"Lighting: {panel.lighting}")
            
        final_prompt = ", ".join(prompt_parts)
        self.logger.debug(f"Final Prompt: {final_prompt}")
        
        return final_prompt
