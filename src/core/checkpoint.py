from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.core.models import ComicScript, Character

class PipelineState(BaseModel):
    """
    Represents the current state of a comic generation pipeline.
    Used for checkpointing and resumption.
    """
    input_hash: str = Field(..., description="Blake2b hash of the input file to ensure consistency")
    last_chunk_index: int = Field(-1, description="Last index of the script chunk successfully processed")
    master_script: Optional[ComicScript] = Field(None, description="The accumulated comic script")
    characters: List[Character] = Field(default_factory=list, description="List of designed characters")
    last_scene_id: int = Field(-1, description="ID of the last scene successfully produced")
    finished_pages: List[str] = Field(default_factory=list, description="List of generated page file paths")
    scene_plans: Dict[int, Any] = Field(default_factory=dict, description="Dictionary of scene plans (id -> plan object)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (timestamps, model names, etc.)")
    
    @property
    def stage(self) -> str:
        """Helper to determine the current stage of the pipeline."""
        if not self.master_script:
            return "scripting"
        if not self.characters:
            return "design"
        return "production"
