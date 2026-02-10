from typing import Any, Dict, List
from src.core.agent import BaseAgent
from src.core.models import Panel, ComicScript
from src.utils.llm_interface import LLMInterface

class DirectorAgent(BaseAgent):
    def __init__(self, agent_name: str = "Director", config: Dict[str, Any] = None):
        super().__init__(agent_name, config)
        self.llm = LLMInterface(model_name=config.get("model_name", "gpt-4o"))

    def process(self, script: ComicScript) -> ComicScript:
        """
        Enhances the script with detailed camera and lighting directions for each panel.
        """
        self.logger.info("Adding directorial guidance to panels...")
        
        system_prompt = """
        You are a visionary film director and cinematographer.
        Your job is to enhance a comic script by adding specific camera angles, framing, and lighting instructions to each panel.
        
        For each panel, suggest:
        1. Camera Angle: (e.g., Low angle, High angle, Bird's eye view, Dutch angle).
        2. Shot Type: (e.g., Close-up, Medium shot, Long shot).
        3. Lighting: (e.g., Dramatic shadows, Soft natural light, Neon backlight).
        
        Update the panel descriptions to include these details naturally.
        """
        
        # We process scene by scene to maintain context
        for scene in script.scenes:
            scene_context = f"Scene: {scene.location}. Summary: {scene.narrative_summary}"
            panels_context = "\n".join([f"Panel {p.id}: {p.description}" for p in scene.panels])
            
            user_prompt = f"""
            Enhance the following scene with cinematic direction:
            {scene_context}
            
            Panels:
            {panels_context}
            
            Return the updated list of panels with enriched descriptions.
            """
            
            try:
                # We expect a list of Panels back
                from pydantic import BaseModel
                class PanelList(BaseModel):
                    panels: List[Panel]

                enhanced_panels = self.llm.generate_structured_output(
                   prompt=user_prompt,
                   system_prompt=system_prompt,
                   schema=PanelList
                )
                
                # Replace the old panels with the new ones
                # Note: This assumes the LLM returns the same number of panels in order.
                # Production code would need robust matching logic.
                if len(enhanced_panels.panels) == len(scene.panels):
                    scene.panels = enhanced_panels.panels
                else:
                    self.logger.warning(f"Director returned {len(enhanced_panels.panels)} panels, expected {len(scene.panels)}. Skipping update for scene {scene.id}.")
            
            except Exception as e:
                 self.logger.error(f"Director failed for scene {scene.id}: {e}")
                 # Continue to next scene rather than crashing entire script
                 continue
                 
        return script
