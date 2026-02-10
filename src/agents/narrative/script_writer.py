from typing import Any, Dict, List
from src.core.agent import BaseAgent
from src.core.models import ComicScript
from src.utils.llm_interface import LLMInterface

class ScriptWriterAgent(BaseAgent):
    def __init__(self, agent_name: str = "ScriptWriter", config: Dict[str, Any] = None):
        super().__init__(agent_name, config)
        self.llm = LLMInterface(model_name=config.get("model_name", "gpt-4o"))

    def process(self, input_text: str) -> ComicScript:
        """
        Converts raw text into a structured ComicScript.
        """
        self.logger.info("Generating comic script from text...")
        
        system_prompt = """
        You are an expert comic book writer. Adapt the provided story into a structured comic book script.
        
        Guidelines:
        - Breakdown the story into logically sequentially scenes and panels.
        - Provide vivid visual descriptions for each panel.
        - Ensure character names and descriptions are consistent throughout.
        - Keep dialogue concise and punchy.
        
        Example Output Format:
        {
          "title": "The Boy and the Sun",
          "synopsis": "A story about a desert shepherd...",
          "scenes": [
            {
              "id": 1,
              "location": "Desert Oasis",
              "narrative_summary": "Santiago wakes up...",
              "characters": [{"name": "Santiago", "description": "Young boy", "personality": "Curious"}],
              "panels": [...]
            }
          ]
        }
        """
        
        user_prompt = f"""
        Adapt the following text into a comic book script:
        
        {input_text}
        """
        
        try:
            script = self.llm.generate_structured_output(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=ComicScript
            )
            return script
        except Exception as e:
            self.logger.error(f"Failed to generate script: {e}")
            raise
