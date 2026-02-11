import unittest
import os
import shutil
import tempfile
from src.core.checkpoint import PipelineState
from src.utils.checkpoint_manager import CheckpointManager
from src.core.storage import LocalStorage
from src.core.models import ComicScript

class TestCheckpointSystem(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage = LocalStorage()
        self.manager = CheckpointManager(self.storage, checkpoint_dir=os.path.join(self.test_dir, ".checkpoints"))
        
        # Mock GitAutomationAgent to prevent it from trying to run git commands in tests
        from unittest.mock import patch
        self.patcher = patch('src.agents.infrastructure.git_automation.GitAutomationAgent')
        self.mock_git = self.patcher.start()
        
        # Create a dummy input file
        self.input_file = os.path.join(self.test_dir, "input.txt")
        with open(self.input_file, "w") as f:
            f.write("Test story content")
            
        self.input_hash = self.manager.get_input_hash(self.input_file)

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_hashing(self):
        """Verify that hashing is consistent for the same content."""
        h1 = self.manager.get_input_hash(self.input_file)
        h2 = self.manager.get_input_hash(self.input_file)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 128) # Blake2b is 128 chars in hex

    def test_save_load_checkpoint(self):
        """Verify that state is preserved through save/load cycle."""
        state = PipelineState(
            input_hash=self.input_hash,
            last_chunk_index=2,
            master_script=ComicScript(title="Test", synopsis="Syn", scenes=[])
        )
        
        self.manager.save_checkpoint(state)
        
        loaded_state = self.manager.load_checkpoint(self.input_hash)
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state.input_hash, self.input_hash)
        self.assertEqual(loaded_state.last_chunk_index, 2)
        self.assertEqual(loaded_state.master_script.title, "Test")

    def test_resume_different_story(self):
        """Verify that a different story doesn't load a mismatched checkpoint."""
        # Save checkpoint for story A
        state_a = PipelineState(input_hash=self.input_hash, last_chunk_index=5)
        self.manager.save_checkpoint(state_a)
        
        # Try loading for story B
        loaded_b = self.manager.load_checkpoint("some_other_hash")
        self.assertIsNone(loaded_b)

    def test_stage_property_logic(self):
        """Verify the stage calculation logic in PipelineState."""
        state = PipelineState(input_hash="hash")
        self.assertEqual(state.stage, "scripting")
        
        state.master_script = ComicScript(title="T", synopsis="S", scenes=[])
        self.assertEqual(state.stage, "design")
        
        # Note: Production stage requires characters list to be non-empty
        from src.core.models import Character
        state.characters = [Character(name="M", description="D", personality="P", pronouns="They/Them")]
        self.assertEqual(state.stage, "production")

if __name__ == "__main__":
    unittest.main()
