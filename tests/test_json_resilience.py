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

    def test_case_insensitivity(self):
        """Verify that 'Title' is mapped to 'title'."""
        bad_json = json.dumps({
            "Title": "Case Test",
            "synopsis": "Test",
            "scenes": []
        })
        repaired = self.agent.repair_json(bad_json, ComicScript)
        self.assertIn("title", repaired)
        self.assertEqual(repaired["title"], "Case Test")

if __name__ == "__main__":
    unittest.main()
