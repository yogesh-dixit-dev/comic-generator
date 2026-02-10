from typing import Any, Dict, List
from src.core.agent import BaseAgent
from src.core.models import Character, CritiqueResult
from src.utils.llm_interface import LLMInterface

class CharacterCritiqueAgent(BaseAgent):
    def __init__(self, agent_name: str = "CharacterCritique", config: Dict[str, Any] = None):
        super().__init__(agent_name, config)
        self.llm = LLMInterface(model_name=config.get("model_name", "gpt-4o"))

    def process(self, characters: List[Character]) -> CritiqueResult:
        """
        Critiques the generated character profiles.
        """
        self.logger.info("Critiquing character designs...")
        
        system_prompt = """
        You represent the Art Director.
        Review the character profiles for:
        1. Visual Distinctiveness: Are the characters easily distinguishable?
        2. Detail: Is there enough physical description for an artist to draw them consistently?
        3. Personality Match: Do the visuals reflect the personality?
        
        Provide specific feedback on how to improve the descriptions.
        """
        
        # Serialize simply for prompt
        char_descriptions = "\n".join([f"- {c.name}: {c.description} (Personality: {c.personality})" for c in characters])
        
        user_prompt = f"""
        Review the following character lineup:
        {char_descriptions}
        
        Are these descriptions robust enough for consistent generation?
        """
        
        try:
            critique = self.llm.generate_structured_output(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=CritiqueResult
            )
            return critique
        except Exception as e:
            self.logger.error(f"Failed to critique characters: {e}")
            raise
