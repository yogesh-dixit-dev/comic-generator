from typing import Any, Dict, List
from src.core.agent import BaseAgent
from src.core.models import Panel, ComicScript
from src.utils.llm_interface import LLMInterface

class DirectorAgent(BaseAgent):
    def __init__(self, agent_name: str = "Director", config: Dict[str, Any] = None):
        super().__init__(agent_name, config)
        self.llm = LLMInterface(model_name=config.get("model_name", "gpt-4o"))

    def process(self, scene: Any) -> Any:
        """
        Enhances a single scene with detailed camera and lighting directions for each panel.
        """
        from src.core.models import Scene, Panel
        
        # Ensure we are working with a Scene object
        if not isinstance(scene, Scene):
            self.logger.warning(f"Director received {type(scene)}, expected Scene. Attempting to skip.")
            return scene

        self.logger.info(f"Adding directorial guidance to Scene {scene.id}...")
        
        system_prompt = """
        You are a visionary film director and cinematographer.
        Your job is to enhance a comic script by adding specific camera angles, framing, and lighting instructions to each panel.
        
        For each panel, suggest:
        1. Camera Angle: (e.g., Low angle, High angle, Bird's eye view, Dutch angle).
        2. Shot Type: (e.g., Close-up, Medium shot, Long shot).
        3. Lighting: (e.g., Dramatic shadows, Soft natural light, Neon backlight).
        
        Return the updated list of panels with enriched descriptions.
        """
        
        scene_context = f"Scene: {scene.location}. Summary: {scene.narrative_summary}"
        panels_context = "\n".join([f"Panel {p.id}: {p.description}" for p in scene.panels])
        
        user_prompt = f"""
        Enhance the following scene with cinematic direction:
        {scene_context}
        
        Panels:
        {panels_context}
        
        Return the updated list of panels.
        """
        
        try:
            from pydantic import BaseModel
            class PanelList(BaseModel):
                panels: List[Panel]

            enhanced_output = self.llm.generate_structured_output(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=PanelList
            )
            
            # Replace the old panels with the new ones
            if len(enhanced_output.panels) == len(scene.panels):
                scene.panels = enhanced_output.panels
                self.logger.info(f"Successfully enhanced {len(scene.panels)} panels for Scene {scene.id}")
            else:
                self.logger.warning(f"Director returned {len(enhanced_output.panels)} panels, expected {len(scene.panels)}. Keeping originals.")
        
        except Exception as e:
            self.logger.error(f"Director failed for scene {scene.id}: {e}")
            
        return scene
