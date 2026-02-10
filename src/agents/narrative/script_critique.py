from typing import Any, Dict
from src.core.agent import BaseAgent
from src.core.models import ComicScript, CritiqueResult
from src.utils.llm_interface import LLMInterface

class ScriptCritiqueAgent(BaseAgent):
    def __init__(self, agent_name: str = "ScriptCritique", config: Dict[str, Any] = None):
        super().__init__(agent_name, config)
        self.llm = LLMInterface(model_name=config.get("model_name", "gpt-4o"))

    def process(self, script: ComicScript) -> CritiqueResult:
        """
        Critiques the generated ComicScript.
        """
        self.logger.info("Critiquing comic script...")
        
        system_prompt = """
        You are a harsh but fair comic book editor.
        Your job is to review a comic script for:
        1. Visual Feasibility: Can the panels be drawn clearly? Are they too cluttered?
        2. Pacing: Does the story flow well?
        3. Dialogue: Is it natural and fits the genre?
        
        Provide specific feedback and a pass/fail judgment.
        """
        
        user_prompt = f"""
        Review the following comic script:
        Title: {script.title}
        Synopsis: {script.synopsis}
        Scenes: {len(script.scenes)}
        
        (Full script content is available in the object structure, providing summary evaluation based on structure)
        
        Evaluate the overall quality and feasibility.
        """
        
        try:
            critique = self.llm.generate_structured_output(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=CritiqueResult
            )
            return critique
        except Exception as e:
            self.logger.error(f"Failed to critique script: {e}")
            raise

    def critique(self, input_data: Any, output_data: Any) -> Dict[str, Any]:
        """
        Override the default critique behavior if needed, but here process IS the critique.
        """
        return {"passed": True, "feedback": "Self-critique not applicable for Critique Agent"}
