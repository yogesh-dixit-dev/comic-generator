from typing import List, Dict, Any
from src.core.agent import BaseAgent
from src.core.models import Character, ComicScript
from src.utils.llm_interface import LLMInterface

class CharacterDesignAgent(BaseAgent):
    def __init__(self, agent_name: str = "CharacterDesigner", config: Dict[str, Any] = None):
        super().__init__(agent_name, config)
        self.llm = LLMInterface(model_name=config.get("model_name", "gpt-4o"))

    def process(self, script: ComicScript) -> List[Character]:
        """
        Analyzes the script and generates detailed character profiles.
        """
        self.logger.info("Generating character design profiles...")
        
        system_prompt = """
        You are a master character designer for comics and animation.
        Your task is to create detailed visual profiles for each main character in the script.
        
        For each character, specify:
        1. Appearance: Physical traits (hair, eyes, build, distinct features).
        2. Attire: Default outfit.
        3. Personality: How their personality affects their look (posture, expression).
        
        The output must be a list of Character objects.
        """
        
        # Extract character names from the script scenes to guide the LLM
        character_names = set()
        for scene in script.scenes:
            if scene.characters:
                for char in scene.characters:
                    character_names.add(char.name)
            if scene.panels:
                 for panel in scene.panels:
                     if panel.characters_present:
                         for name in panel.characters_present:
                             character_names.add(name)

        user_prompt = f"""
        Analyze the following script and create character profiles for these characters: {', '.join(character_names)}
        
        Script Title: {script.title}
        Script Synopsis: {script.synopsis}
        
        (Infer details from the context of the story if not explicitly stated).
        """
        
        # We need a wrapper model because the LLM returns a single object, but we want a list
        from pydantic import BaseModel
        class CharacterList(BaseModel):
            characters: List[Character]
        
        try:
            result = self.llm.generate_structured_output(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=CharacterList
            )
            return result.characters
        except Exception as e:
            self.logger.error(f"Failed to generate character profiles: {e}")
            raise
