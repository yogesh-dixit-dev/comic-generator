import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from src.core.models import ComicScript, ScenePlan

class TestPipelineIsolation(unittest.TestCase):
    def setUp(self):
        # Mocking arguments
        self.args = MagicMock()
        self.args.input = "test.txt"
        self.args.output = "output"
        self.args.storage = "local"
        self.args.reasoning_model = "ollama/llama3.1"
        self.args.fast_model = "ollama/llama3.2"
        self.args.colab = True
        self.args.hf_repo = ""

    @patch('src.utils.llm_interface.LLMInterface')
    @patch('src.main.DirectorAgent')
    @patch('src.main.DiffusersImageGenerator')
    @patch('src.main.CheckpointManager')
    def test_phase_plan_logic(self, mock_checkpoint, mock_gen, mock_director, mock_llm):
        """Verify that 'plan' phase runs Director but NOT Image Generator."""
        from src.main import main
        
        self.args.phase = "plan"
        
        # Setup mocks
        mock_llm.return_value.is_healthy.return_value = True
        mock_checkpoint.return_value.load_checkpoint.return_value = None
        mock_director.return_value.run.return_value = ScenePlan(id=1, location="Test", narrative_summary="Syn", panels=[])
        
        # We need to mock input reading too
        with patch('src.main.InputReaderAgent.process', return_value="Test story"):
            with patch('src.main.ScriptWriterAgent.run', return_value=ComicScript(title="T", synopsis="S", scenes=[])):
                # Mock sys.argv to avoid catching real args if run in some environments
                with patch('argparse.ArgumentParser.parse_args', return_value=self.args):
                    main()
        
        # Assertions
        mock_director.return_value.run.assert_called()
        mock_gen.assert_not_called()
        mock_llm.return_value.unload_model.assert_not_called() # Only happens in draw or all

    @patch('src.utils.llm_interface.LLMInterface')
    @patch('src.main.IllustratorAgent')
    @patch('src.main.DiffusersImageGenerator')
    @patch('src.main.CheckpointManager')
    def test_phase_draw_logic(self, mock_checkpoint, mock_gen, mock_illustrator, mock_llm):
        """Verify that 'draw' phase skips health check and runs Illustrator."""
        from src.main import main
        
        self.args.phase = "draw"
        
        # Setup mocks to simulate existing plan
        mock_state = MagicMock()
        mock_state.scene_plans = {"1": ScenePlan(id=1, location="Test", narrative_summary="Syn", panels=[]).dict()}
        mock_checkpoint.return_value.load_checkpoint.return_value = mock_state
        
        with patch('src.main.InputReaderAgent.process', return_value="Test story"):
            with patch('src.main.ScriptWriterAgent.run', return_value=ComicScript(title="T", synopsis="S", scenes=[])):
                with patch('argparse.ArgumentParser.parse_args', return_value=self.args):
                     main()
        
        # Assertions
        mock_llm.return_value.is_healthy.assert_not_called() # Skipped in draw
        mock_illustrator.return_value.run_batch.assert_called()

if __name__ == "__main__":
    unittest.main()
