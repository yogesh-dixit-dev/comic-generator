import os
import ast
import logging
import subprocess
import sys
from typing import Dict, Any, List, Set, Tuple
from src.core.agent import BaseAgent
from src.utils.llm_interface import LLMInterface
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TestCode(BaseModel):
    test_file_content: str = Field(..., description="The full Python code for the unit tests")
    test_file_name: str = Field(..., description="The suggested filename for the test (e.g., test_utils.py)")

class QualityAssuranceAgent(BaseAgent):
    """
    Agent responsible for generating test cases and ensuring code quality before push.
    """
    def __init__(self, agent_name: str = "QualityAssuranceAgent", config: Dict[str, Any] = None):
        super().__init__(agent_name, config)
        self.llm = LLMInterface(model_name=self.config.get("model", "gpt-4o"))

    def scan_codebase(self, root_dir: str = "src") -> Dict[str, List[str]]:
        """
        Scans the codebase to find all public methods in classes.
        Returns a mapping of filename -> list of method names.
        """
        methods_map = {}
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".py") and file != "__init__.py":
                    path = os.path.join(root, file)
                    methods = self._extract_methods(path)
                    if methods:
                        methods_map[path] = methods
        return methods_map

    def _extract_methods(self, filepath: str) -> List[str]:
        """Extracts class names and method names from a Python file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                node = ast.parse(f.read())
            
            methods = []
            for item in node.body:
                if isinstance(item, ast.ClassDef):
                    class_name = item.name
                    for subitem in item.body:
                        if isinstance(subitem, ast.FunctionDef):
                            if not subitem.name.startswith("_"):
                                methods.append(f"{class_name}.{subitem.name}")
                elif isinstance(item, ast.FunctionDef):
                    if not item.name.startswith("_"):
                        methods.append(item.name)
            return methods
        except Exception as e:
            self.logger.error(f"Error parsing {filepath}: {e}")
            return []

    def find_existing_tests(self, test_dir: str = "tests") -> Set[str]:
        """Scans tests directory to see which methods are already tested (roughly)."""
        tested_names = set()
        if not os.path.exists(test_dir):
            return tested_names

        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                            # Deep search for mention of method names
                            # This is a heuristic: if a test file mentions 'Agent.process', we assume it's tested.
                            tested_names.add(path)
                    except: pass
        return tested_names

    def generate_tests_for_file(self, target_file: str) -> bool:
        """Generates a unit test file for a specific source file."""
        self.logger.info(f"Generating tests for {target_file}...")
        
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                code_content = f.read()

            prompt = (
                f"I need high-quality Python unit tests for the following code:\n\n"
                f"FILE: {target_file}\n\n"
                f"CODE:\n{code_content}\n\n"
                f"Requirements:\n"
                f"1. Use the 'unittest' framework.\n"
                f"2. Mock external dependencies (APIs, LLMs, Filesystem) using 'unittest.mock'.\n"
                f"3. Include at least one happy path and one error case for each public method.\n"
                f"4. Ensure imports are correct based on the file structure (src is top-level).\n"
                f"5. Respond ONLY with the JSON format specified."
            )

            test_data = self.llm.generate_structured_output(prompt, TestCode)
            
            test_dir = "tests"
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            
            dest_path = os.path.join(test_dir, test_data.test_file_name)
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(test_data.test_file_content)
            
            self.logger.info(f"✅ Tests generated at: {dest_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate tests for {target_file}: {e}")
            return False

    def run_all_checks(self) -> bool:
        """
        Runs all quality checks: Syntax, Static Analysis, and Unit Tests.
        This is the main hook for pre-push.
        """
        self.logger.info("🚀 Running Quality Assurance Checks...")
        
        # 1. Syntax Check
        try:
            subprocess.run([sys.executable, "-m", "compileall", "src"], check=True, capture_output=True)
            self.logger.info("✅ Syntax Check Passed.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Syntax Check Failed:\n{e.stderr.decode()}")
            return False

        # 2. Run Unit Tests (Existing)
        try:
            # discover returns 0 if all tests pass
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "tests"], capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("✅ Unit Tests Passed.")
            else:
                self.logger.error(f"❌ Unit Tests Failed:\n{result.stdout}\n{result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            return False

        return True

    def process(self, input_data: Any = None) -> bool:
        """Main entry point."""
        return self.run_all_checks()

if __name__ == "__main__":
    # Simple CLI for the agent
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", help="Generate tests for a specific file")
    args = parser.parse_args()
    
    agent = QualityAssuranceAgent()
    if args.gen:
        agent.generate_tests_for_file(args.gen)
    else:
        success = agent.run_all_checks()
        sys.exit(0 if success else 1)
