import unittest
from src.agents.visual.consistency_manager import ConsistencyManager
from src.core.models import Character, Panel

class TestPromptCompression(unittest.TestCase):
    def setUp(self):
        self.manager = ConsistencyManager()

    def test_short_prompt_no_compression(self):
        """Verify that short prompts are not modified."""
        panel = Panel(
            id=1,
            description="A simple scene",
            characters_present=["Santiago"],
            dialogue=[]
        )
        characters = [
            Character(name="Santiago", description="Young shepherd", personality="Quiet", pronouns="He/Him")
        ]
        prompt = self.manager.process(panel, characters)
        self.assertIn("Young shepherd", prompt)
        self.assertIn("Quiet", prompt)

    def test_long_prompt_compression(self):
        """Verify that prompts within 225 tokens are NOT truncated, but extreme ones ARE."""
        panel = Panel(
            id=2,
            description="A complex scene",
            characters_present=["CharacterA"],
            camera_angle="Wide shot",
            lighting="Dramatic shadows",
            dialogue=[]
        )
        # 800 characters - should stay within 225 tokens (~900 chars)
        long_desc = "This is a detailed description" * 20 
        characters = [
            Character(name="CharacterA", description=long_desc, personality="Very layered personality", pronouns="They/Them")
        ]
        
        prompt = self.manager.process(panel, characters)
        # Should NOT be truncated
        self.assertIn("Very layered personality", prompt)
        self.assertLessEqual(len(prompt), 1000)

    def test_extreme_prompt_compression(self):
        """Verify that prompts exceeding 225 tokens are truncated."""
        panel = Panel(id=3, description="Extreme scene", characters_present=["X"], dialogue=[])
        # 1200+ characters - should definitely trigger truncation
        extreme_desc = "unnecessary detail " * 100
        characters = [Character(name="X", description=extreme_desc, personality="Complex", pronouns="It")]
        
        prompt = self.manager.process(panel, characters)
        # Should be truncated
        self.assertLessEqual(len(prompt), 1000) # Our char_limit is 225 * 4 = 900, with some margin
        self.assertIn("...", prompt)

if __name__ == "__main__":
    unittest.main()
