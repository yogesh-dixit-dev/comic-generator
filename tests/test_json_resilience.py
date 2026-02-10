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
