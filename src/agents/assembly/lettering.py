from typing import Any, Dict
from src.core.agent import BaseAgent
from src.core.models import Panel
from PIL import Image, ImageDraw, ImageFont # Using Pillow for simple rendering
import os

class LetteringAgent(BaseAgent):
    def process(self, panel: Panel) -> str:
        """
        Adds text/speech bubbles to the panel image.
        """
        self.logger.info(f"Adding lettering to Panel {panel.id}...")
        
        if not panel.image_path or not os.path.exists(panel.image_path):
             self.logger.warning(f"No image found for Panel {panel.id}, skipping lettering.")
             return None
             
        try:
            # Load the image
            img = Image.open(panel.image_path)
            draw = ImageDraw.Draw(img)
            
            # Simple text overlay logic (bottom of image)
            # In a real app, uses bubble localization algorithms or user manual placement
            for dialogue in panel.dialogue:
                 speaker = dialogue.get('speaker', 'Unknown')
                 text = dialogue.get('text', '')
                 full_text = f"{speaker}: {text}"
                 
                 # Draw text at bottom
                 # (Pseudo-code for positioning)
                 text_pos = (10, img.height - 40)
                 draw.text(text_pos, full_text, fill="white")
            
            # Save the new version
            output_path = panel.image_path.replace(".png", "_lettered.png")
            img.save(output_path)
            return output_path
            
        except Exception as e:
            self.logger.error(f"Lettering failed: {e}")
            return panel.image_path
