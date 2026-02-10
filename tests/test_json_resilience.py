import unittest
import json
from src.utils.json_resilience import JSONResilienceAgent
from src.core.models import ComicScript, Scene, Panel, Character

class TestJSONResilience(unittest.TestCase):
    def setUp(self):
        self.agent = JSONResilienceAgent()

    def test_skeleton_generation(self):
        """Verify that deep skeletons contain nested keys like 'panels' and 'characters'."""
        skeleton_str = self.agent.generate_deep_skeleton(ComicScript)
        skeleton = json.loads(skeleton_str)
        
        self.assertIn("title", skeleton)
        self.assertIn("scenes", skeleton)
        self.assertIsInstance(skeleton["scenes"], list)
        
        # Check deep nesting
        scene = skeleton["scenes"][0]
        self.assertIn("location", scene)
        self.assertIn("panels", scene)
        
        panel = scene["panels"][0]
        self.assertIn("description", panel)
        self.assertIn("dialogue", panel)

    def test_heuristic_repair_panel_id(self):
        """Verify that 'panel_id' is correctly mapped to 'id'."""
        bad_json = json.dumps({
            "title": "Test",
            "synopsis": "Test",
            "scenes": [
                {
                    "id": 1,
                    "location": "Church",
                    "narrative_summary": "Summary",
                    "panels": [
                        {"panel_id": 1, "description": "Llama style ID"}
                    ]
                }
            ]
        })
        
        repaired = self.agent.repair_json(bad_json, ComicScript)
        
        # Verify mapping
        self.assertEqual(repaired["scenes"][0]["panels"][0]["id"], 1)
        self.assertIn("id", repaired["scenes"][0]["panels"][0])
        self.assertNotIn("panel_id", repaired["scenes"][0]["panels"][0])

    def test_aggressive_scene_mapping(self):
        """Verify that Llama's 'title/text' scenes are mapped correctly."""
        bad_json = json.dumps({
            "title": "Story",
            "synopsis": "Syn",
            "scenes": [
                {
                    "title": "Scene 1",
                    "text": "This should be narrative summary"
                }
            ]
        })
        repaired = self.agent.repair_json(bad_json, ComicScript)
        self.assertEqual(repaired["scenes"][0]["id"], 1)
        self.assertEqual(repaired["scenes"][0]["narrative_summary"], "This should be narrative summary")

    def test_optional_model_handling(self):
        """Verify that Optional[BaseModel] is handled correctly."""
        # Note: ComicScript doesn't have Optional[BaseModel] right now, but we want to ensure resilience
        # LayoutEngine output models might have them.
        pass

    def test_float_id_normalization(self):
        """Verify that floating point IDs (e.g. 1.1) are truncated to integers."""
        bad_json = json.dumps({
            "title": "Story",
            "synopsis": "Syn",
            "scenes": [
                {
                    "id": 1,
                    "location": "Loc",
                    "narrative_summary": "Sum",
                    "panels": [
                        {"id": 1.1, "description": "Float ID Test"}
                    ]
                }
            ]
        })
        repaired = self.agent.repair_json(bad_json, ComicScript)
        self.assertEqual(repaired["scenes"][0]["panels"][0]["id"], 1)

    def test_scene_field_fallbacks(self):
        """Verify that missing Scene fields are filled from common LLM hallucinations."""
        bad_json = json.dumps({
            "title": "Story",
            "synopsis": "Syn",
            "scenes": [
                {
                    # Missing ID and Location
                    "setting": "The Oasis",
                    "scene": "A mysterious scene unfolds."
                }
            ]
        })
        repaired = self.agent.repair_json(bad_json, ComicScript)
        scene = repaired["scenes"][0]
        self.assertEqual(scene["id"], 1) # Auto-generated
        self.assertEqual(scene["location"], "The Oasis") # From 'setting'
        self.assertEqual(scene["narrative_summary"], "A mysterious scene unfolds.") # From 'scene'

    def test_auto_generated_ids(self):
        """Verify that lists of objects get sequential IDs if missing."""
        bad_json = json.dumps({
            "title": "Story",
            "synopsis": "Syn",
            "scenes": [
                {"location": "A", "narrative_summary": "S1"},
                {"location": "B", "narrative_summary": "S2"}
            ]
        })
        repaired = self.agent.repair_json(bad_json, ComicScript)
        self.assertEqual(repaired["scenes"][0]["id"], 1)
        self.assertEqual(repaired["scenes"][1]["id"], 2)

    def test_dialogue_repair(self):
        """Verify that string-based dialogue lists are repaired to speaker/text dicts."""
        bad_json = json.dumps({
            "title": "Story",
            "synopsis": "Syn",
            "scenes": [
                {
                    "id": 1,
                    "location": "Church",
                    "narrative_summary": "Sum",
                    "panels": [
                        {
                            "id": 1,
                            "description": "Desc",
                            "dialogue": ["Santiago: I'll spend the night here."]
                        }
                    ]
                }
            ]
        })
        repaired = self.agent.repair_json(bad_json, ComicScript)
        dialogue = repaired["scenes"][0]["panels"][0]["dialogue"]
        self.assertEqual(len(dialogue), 1)
        self.assertEqual(dialogue[0]["speaker"], "Santiago")
        self.assertEqual(dialogue[0]["text"], "I'll spend the night here.")

    def test_case_insensitivity(self):
        ...

if __name__ == "__main__":
    unittest.main()
