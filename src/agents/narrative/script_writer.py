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
        You are an expert comic book writer and director. 
        Your task is to adapt the provided story text into a detailed comic book script.
        
        Focus on:
        1. Visual storytelling: Show, don't just tell.
        2. Pacing: Break down the story into scenes and panels effectively.
        3. Character consistency: Ensure character descriptions are consistent.
        4. Dialogue: Keep it punchy and suitable for comics.
        
        Output must be a valid JSON object matching the ComicScript schema.
        
        Example structure:
        {
          "title": "Story Title",
          "synopsis": "A brief summary...",
          "scenes": [
            {
              "id": 1,
              "location": "Forest",
              "narrative_summary": "Introduction...",
              "characters": [{"name": "Hero", "description": "Tall", "personality": "Brave"}],
              "panels": [
                {
                  "id": 1,
                  "description": "Hero enters the woods",
                  "dialogue": [{"speaker": "Hero", "text": "It's dark here."}],
                  "characters_present": ["Hero"]
                }
              ]
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
