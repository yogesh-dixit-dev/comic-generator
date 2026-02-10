from typing import List
from src.core.models import ComicScript, Scene

class ScriptConsolidator:
    def __init__(self):
        self.master_script: ComicScript = None
        self.total_scenes = 0

    def add_chunk(self, script: ComicScript):
        if self.master_script is None:
            self.master_script = script
            self.total_scenes = len(script.scenes)
        else:
            # Append scenes and update IDs
            for scene in script.scenes:
                self.total_scenes += 1
                scene.id = self.total_scenes
                self.master_script.scenes.append(scene)
            
            # Optionally update title/synopsis if needed, though usually the first chunk is best
            if not self.master_script.synopsis and script.synopsis:
                self.master_script.synopsis = script.synopsis

    def get_script(self) -> ComicScript:
        return self.master_script
