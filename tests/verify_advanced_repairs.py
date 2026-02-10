import sys
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Add src to path
sys.path.append(os.getcwd())

from src.utils.json_resilience import JSONResilienceAgent
from src.core.models import ComicScript

def test_hallucination_repairs():
    agent = JSONResilienceAgent()
    
    # 📢 SIMULATED HALLUCINATION: Llama 3.1 providing lists for strings and strings for objects
    hallucinated_data = {
        "title": "The Alchemist",
        "synopsis": "A shepherd boy travels.",
        "scenes": [
            {
                "id": 1,
                "location": "Tarija",
                "narrative_summary": "Santiago meets the King.",
                "characters": [
                    {
                        "name": "Santiago",
                        "pronouns": ["He", "Him"], # ❌ LIST INSTEAD OF STRING
                        "description": "Young shepherd",
                        "personality": "Adventurous"
                    }
                ],
                "panels": [
                    {
                        "id": 1,
                        "description": "Santiago looks at the sky.",
                        "dialogue": [
                            "I'll spend the night here.", # ❌ STRING INSTEAD OF DICT
                            "Narrator: The wind blows." # ❌ STRING INSTEAD OF DICT
                        ],
                        "characters_present": ["Santiago"]
                    }
                ]
            }
        ]
    }

    print("--- 🧪 Testing Advanced Repair Logic ---")
    repaired = agent.repair_json(hallucinated_data, ComicScript)
    
    # Verify Pronouns (List -> String)
    repaired_char = repaired["scenes"][0]["characters"][0]
    print(f"Pronouns Repaired: {repaired_char['pronouns']}")
    assert isinstance(repaired_char['pronouns'], str)
    assert repaired_char['pronouns'] == "He/Him"

    # Verify Dialogue (String -> Dict)
    repaired_dialogue = repaired["scenes"][0]["panels"][0]["dialogue"]
    print(f"Dialogue Repaired: {repaired_dialogue}")
    assert isinstance(repaired_dialogue[0], dict)
    assert repaired_dialogue[0]["speaker"] == "Narrator"
    assert repaired_dialogue[1]["speaker"] == "Narrator"

    print("\n✅ Advanced Repair Verification Passed!")

if __name__ == "__main__":
    try:
        test_hallucination_repairs()
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()
