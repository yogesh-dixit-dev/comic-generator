import json
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from src.utils.llm_interface import LLMInterface

# Mock schemas to match models.py
class Character(BaseModel):
    name: str
    description: str

class Panel(BaseModel):
    id: int
    description: str
    dialogue: List[Dict[str, str]]

class Scene(BaseModel):
    id: int
    location: str
    panels: List[Panel]

class ComicScript(BaseModel):
    title: str
    scenes: List[Scene]

def test_extraction_robustness():
    interface = LLMInterface(model_name="ollama/llama3.2")
    
    # Conversational output similar to what was failing
    conversational_text = """
    This is a story about a young shepherd who sets out on a journey...
    
    Here is the JSON script:
    {
      "title": "The Alchemist's Dream",
      "scenes": [
        {
          "id": 1,
          "location": "Spanish Pastures",
          "panels": [
            {
              "id": 1,
              "description": "Santiago looks at the stars.",
              "dialogue": [{"speaker": "Santiago", "text": "I will find my treasure."}]
            }
          ]
        }
      ]
    }
    """
    
    print("Testing extraction from conversational text...")
    extracted = interface._extract_json(conversational_text)
    print(f"Extracted: {extracted[:50]}...")
    
    try:
        data = json.loads(extracted)
        script = ComicScript.model_validate(data)
        print("✅ Validation successful!")
        print(f"Title: {script.title}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")

if __name__ == "__main__":
    test_extraction_robustness()
