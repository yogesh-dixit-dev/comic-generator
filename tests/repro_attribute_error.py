import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.llm_interface import LLMInterface
from src.core.models import CritiqueResult

class TestLLMErrorHandling(unittest.TestCase):
    @patch('src.utils.llm_interface.completion')
    def test_generate_structured_output_raises_on_failure(self, mock_completion):
        # Mock completion to always fail
        mock_completion.side_effect = Exception("Connection refused")
        
        interface = LLMInterface(model_name="ollama/llama3.1")
        
        # This currently returns None, which is the bug
        result = interface.generate_structured_output(
            prompt="test",
            schema=CritiqueResult
        )
        
        if result is None:
            print("FAILURE: generate_structured_output returned None instead of raising")
        else:
            print(f"SUCCESS?: generate_structured_output returned {result}")

if __name__ == "__main__":
    # Just run the test logic directly for quick check
    from src.utils.llm_interface import LLMInterface
    from src.core.models import CritiqueResult
    import litellm
    
    with patch('src.utils.llm_interface.completion') as mock_comp:
        mock_comp.side_effect = Exception("Connection refused")
        interface = LLMInterface(model_name="ollama/llama3.1")
        
        print("Testing generate_structured_output with failing completion...")
        try:
            result = interface.generate_structured_output(
                prompt="test",
                schema=CritiqueResult
            )
            if result is None:
                print("BUG REPRODUCED: result is None")
            else:
                print(f"Unexpected result: {result}")
        except Exception as e:
            print(f"Properly raised exception: {e}")
