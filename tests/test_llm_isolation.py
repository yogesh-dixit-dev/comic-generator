import unittest
from unittest.mock import patch, MagicMock
from src.utils.llm_interface import LLMInterface
import requests

class TestLLMIsolation(unittest.TestCase):
    def setUp(self):
        self.llm = LLMInterface(model_name="ollama/llama3.1")

    @patch('requests.get')
    def test_is_healthy_success(self, mock_get):
        """Verify is_healthy returns True if a host responds with 200."""
        # Mocking 127.0.0.1 success
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.side_effect = [Exception("localhost failed"), mock_resp]
        
        self.assertTrue(self.llm.is_healthy())
        self.assertEqual(mock_get.call_count, 2)

    @patch('requests.get')
    def test_is_healthy_failure(self, mock_get):
        """Verify is_healthy returns False if all hosts fail."""
        mock_get.side_effect = [Exception("fail"), Exception("fail")]
        self.assertFalse(self.llm.is_healthy())
        self.assertEqual(mock_get.call_count, 2)

    @patch('litellm.completion')
    def test_unload_model(self, mock_completion):
        """Verify unload_model sends keep_alive=0."""
        self.llm.unload_model()
        
        # Verify litellm was called with keep_alive=0
        args, kwargs = mock_completion.call_args
        self.assertEqual(kwargs.get("keep_alive"), 0)
        self.assertEqual(kwargs.get("model"), "ollama/llama3.1")

if __name__ == "__main__":
    unittest.main()
