import sys
import os
import json

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.json_resilience import JSONResilienceAgent
from src.core.models import ComicScript

def test_dialogue_repair():
    agent = JSONResilienceAgent()
    
    # Simulate the problematic LLM output where dialogue is a list of strings
    bad_json = {
        "title": "Test Comic",
        "synopsis": "A test synopsis",
        "scenes": [
            {
                "id": 1,
                "location": "Test Location",
                "narrative_summary": "Test Summary",
                "characters": [
                    {
                        "name": "Santiago",
                        "pronouns": "He/Him",
                        "description": "A boy",
                        "personality": "Nice"
                    }
                ],
                "panels": [
                    {
                        "id": 1,
                        "description": "Santiago speaks",
                        "dialogue": ["'I need to spend the night here.'"],
                        "characters_present": ["Santiago"]
                    },
                    {
                        "id": 2,
                        "description": "Another panel",
                        "dialogue": ["Santiago: Wait for me!"],
                        "characters_present": ["Santiago"]
                    }
                ]
            }
        ]
    }
    
    raw_json_str = json.dumps(bad_json)
    print("Repairing JSON...")
    
    repaired_data = agent.repair_json(raw_json_str, ComicScript)
    
    print("\nRepaired Dialogue for Panel 1:")
    print(repaired_data['scenes'][0]['panels'][0]['dialogue'])
    
    print("\nRepaired Dialogue for Panel 2:")
    print(repaired_data['scenes'][0]['panels'][1]['dialogue'])
    
    # Validate with Pydantic
    try:
        script = ComicScript.model_validate(repaired_data)
        print("\n✅ Validation Successful!")
        assert script.scenes[0].panels[0].dialogue[0]['text'] == "I need to spend the night here."
        assert script.scenes[0].panels[0].dialogue[0]['speaker'] == "Narrator"
        
        assert script.scenes[0].panels[1].dialogue[0]['text'] == "Wait for me!"
        assert script.scenes[0].panels[1].dialogue[0]['speaker'] == "Santiago"
        
    except Exception as e:
        print(f"\n❌ Validation Failed: {e}")
        sys.exit(1)

def test_skeleton():
    agent = JSONResilienceAgent()
    skeleton = agent.generate_deep_skeleton(ComicScript)
    print("\nGenerated Skeleton:")
    print(skeleton)
    
    assert '"dialogue": [' in skeleton
    assert '"speaker": "Character Name"' in skeleton

if __name__ == "__main__":
    test_dialogue_repair()
    test_skeleton()
