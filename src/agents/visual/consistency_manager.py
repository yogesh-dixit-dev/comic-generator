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
        
        style = style_guide or self.style_preset
        base_parts = [style, panel.description]
        
        if panel.camera_angle:
            base_parts.append(f"Camera: {panel.camera_angle}")
        if panel.lighting:
            base_parts.append(f"Lighting: {panel.lighting}")

        # Character parts are handled separately for compression
        char_parts = []
        if panel.characters_present:
            for char_name in panel.characters_present:
                character = None
                for c in characters:
                    if c.name.lower() == char_name.lower() or any(alias.lower() == char_name.lower() for alias in c.aliases):
                        character = c
                        break
                
                if character:
                    char_parts.append({
                        "name": character.name,
                        "desc": character.description,
                        "personality": character.personality
                    })

        final_prompt = self._assemble_and_compress(base_parts, char_parts)
        self.logger.debug(f"Final Prompt: {final_prompt}")
        
        return final_prompt

    def _assemble_and_compress(self, base_parts: List[str], char_parts: List[Dict[str, str]], max_tokens: int = 70) -> str:
        """
        Assembles prompt parts and compresses character descriptions if they exceed the token limit.
        Uses a heuristic of 4 characters per token.
        """
        char_limit = max_tokens * 4
        
        # 1. First Pass: Full descriptions
        full_char_strings = [f"({c['name']}: {c['desc']}, {c['personality']} expression)" for c in char_parts]
        combined = ", ".join(base_parts + full_char_strings)
        
        if len(combined) <= char_limit:
            return combined

        # 2. Second Pass: Drop personalities
        self.logger.warning(f"Prompt exceeds ~{max_tokens} tokens. Dropping personality traits...")
        compressed_chars = [f"({c['name']}: {c['desc']})" for c in char_parts]
        combined = ", ".join(base_parts + compressed_chars)
        
        if len(combined) <= char_limit:
            return combined

        # 3. Third Pass: Aggressively truncate individual character descriptions
        # We calculate remaining space for characters
        base_str = ", ".join(base_parts)
        remaining_chars = char_limit - len(base_str) - (len(char_parts) * 5) # 5 for ", ()" overhead
        
        if remaining_chars > 0:
            per_char_limit = remaining_chars // len(char_parts)
            truncated_chars = []
            for c in char_parts:
                char_prompt = f"{c['name']}: {c['desc']}"
                if len(char_prompt) > per_char_limit:
                    char_prompt = char_prompt[:per_char_limit-3] + "..."
                truncated_chars.append(f"({char_prompt})")
            combined = base_str + ", " + ", ".join(truncated_chars)
        else:
            # If base prompt is already too long, we just keep the names
            combined = base_str + ", " + ", ".join([f"({c['name']})" for c in char_parts])

        # 4. Final hard cut safety
        if len(combined) > char_limit:
             return combined[:char_limit]

        return combined
