from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Character(BaseModel):
    name: str = Field(..., description="The unique, canonical name of the character")
    aliases: List[str] = Field(default_factory=list, description="Other names or titles used for this character (e.g., 'The Young Shepherd')")
    pronouns: str = Field(..., description="Pronouns for the character (e.g., 'He/Him')")
    description: str = Field(..., description="Detailed physical description (age, hair, clothes, distinct features)")
    personality: str = Field(..., description="Core personality traits and typical expressions")
    reference_image_path: Optional[str] = Field(None, description="Path to a reference image file if available")

class Panel(BaseModel):
    id: int = Field(..., description="Panel number within the scene")
    description: str = Field(..., description="Detailed visual description of the panel content")
    dialogue: List[Dict[str, str]] = Field(default_factory=list, description="List of dialogue lines, e.g. [{'speaker': 'John', 'text': 'Hello!'}]")
    characters_present: List[str] = Field(default_factory=list, description="List of character names present in this panel")
    camera_angle: Optional[str] = Field(None, description="Suggested camera angle (e.g., 'Close-up', 'Wide shot')")
    lighting: Optional[str] = Field(None, description="Lighting description")
    image_prompt: Optional[str] = Field(None, description="The final prompt sent to the image generator")
    image_path: Optional[str] = Field(None, description="Path to the generated image file")

class Scene(BaseModel):
    id: int = Field(..., description="Scene number")
    location: str = Field(..., description="Setting/Location of the scene")
    characters: List[Character] = Field(default_factory=list, description="Characters involved in this scene")
    panels: List[Panel] = Field(default_factory=list, description="Ordered list of panels in the scene")
    narrative_summary: str = Field(..., description="Summary of the plot points covered in this scene")

class ComicScript(BaseModel):
    title: str = Field(..., description="Title of the comic")
    synopsis: str = Field(..., description="Brief synopsis of the story")
    scenes: List[Scene] = Field(..., description="List of scenes in the comic")
    style_guide: Optional[str] = Field(None, description="Overall artistic style description")

class CritiqueResult(BaseModel):
    passed: bool = Field(..., description="Whether the artifact passed the critique")
    feedback: str = Field(..., description="Detailed feedback on issues found")
    score: float = Field(..., ge=0.0, le=10.0, description="Quality score from 0-10")
