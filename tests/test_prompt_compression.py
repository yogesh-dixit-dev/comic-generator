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
        """Verify that very long prompts are compressed to stay near the token limit (70 tokens * 4 = 280 chars)."""
        panel = Panel(
            id=2,
            description="A complex scene with many details and backgrounds",
            characters_present=["CharacterA", "CharacterB"],
            camera_angle="Wide shot",
            lighting="Dramatic shadows",
            dialogue=[]
        )
        # Create very long descriptions
        long_desc = "This is an extremely long and detailed description of a character that includes many unnecessary details about their clothing, hair color, eye shape, and the way they stand in the sunlight which should definitely trigger truncation."
        characters = [
            Character(name="CharacterA", description=long_desc, personality="Very complex and layered personality that takes many words to describe", pronouns="They/Them"),
            Character(name="CharacterB", description=long_desc, personality="Another very complex personality that adds to the token count significantly", pronouns="They/Them")
        ]
        
        prompt = self.manager.process(panel, characters)
        
        # Heuristic check: length should be around 280-300 chars
        self.assertLessEqual(len(prompt), 320) 
        self.assertIn("CharacterA", prompt)
        self.assertIn("CharacterB", prompt)
        self.assertIn("Wide shot", prompt)
        # Verify personality was likely dropped or description truncated
        self.assertNotIn("layered personality", prompt)

if __name__ == "__main__":
    unittest.main()
