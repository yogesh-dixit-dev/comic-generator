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
            
            # Comic book lettering logic
            # For each dialogue, we draw a bubble. 
            # In Phase 2, we will eventually allow the UI to specify coordinates.
            # For now, we stack them from the top left.
            y_offset = 20
            
            for dialogue in panel.dialogue:
                 speaker = dialogue.get('speaker', 'Unknown')
                 text = dialogue.get('text', '')
                 full_text = f"{speaker.upper()}\n{text}"
                 
                 # Basic font - in production, load a comic font (.ttf)
                 font_size = 18
                 try:
                     # Attempt to load a default font that might be available
                     font = ImageFont.truetype("arial.ttf", font_size)
                 except:
                     font = ImageFont.load_default()

                 # Calculate text size to size the bubble
                 # text_size = draw.multiline_textbbox((0, 0), full_text, font=font)
                 # Simpler estimation for compatibility
                 lines = full_text.split('\n')
                 max_w = max([len(line) for line in lines]) * 10
                 total_h = len(lines) * 20 + 20
                 
                 # Draw Bubble (Ellipse)
                 bubble_rect = [20, y_offset, 20 + max_w + 30, y_offset + total_h + 20]
                 draw.ellipse(bubble_rect, fill="white", outline="black", width=2)
                 
                 # Draw Text
                 draw.multiline_text((40, y_offset + 10), full_text, fill="black", font=font, align="center")
                 
                 y_offset += total_h + 40
            
            # Save the new version
            output_path = panel.image_path.replace(".png", "_lettered.png")
            img.save(output_path)
            return output_path
            
        except Exception as e:
            self.logger.error(f"Lettering failed: {e}")
            return panel.image_path
