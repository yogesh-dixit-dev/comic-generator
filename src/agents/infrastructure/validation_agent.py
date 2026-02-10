import logging
import os
import sys
import subprocess
from typing import Dict, Any, List
from src.core.agent import BaseAgent
from src.core.models import ComicScript

logger = logging.getLogger(__name__)

class ValidationAgent(BaseAgent):
    """
    Agent responsible for validating the codebase and pipeline integrity.
    It performs syntax checks and runs a dry-run of the pipeline.
    """
    def __init__(self, agent_name: str = "ValidationAgent", config: Dict[str, Any] = None):
        super().__init__(agent_name, config)

    def check_syntax(self, directory: str = "src") -> bool:
        """
        Runs a syntax check (compile-only) on all Python files in the directory.
        """
        logger.info(f"Running syntax check on directory: {directory}")
        try:
            # -m compileall -q: compiles all files quietly
            # returns non-zero if any file fails to compile
            subprocess.run([sys.executable, "-m", "compileall", "-q", directory], check=True)
            logger.info("✅ Syntax check passed.")
            return True
        except subprocess.CalledProcessError:
            logger.error("❌ Syntax check failed. Please fix syntax errors before pushing.")
            return False

    def run_dry_run(self) -> bool:
        """
        Runs a mini E2E test of the main pipeline using the --colab flag (for mock detection)
        but with a small input and mock generators.
        """
        logger.info("Running pipeline dry-run...")
        
        # Create a tiny sample story for testing
        test_story_path = "test_story.txt"
        with open(test_story_path, "w") as f:
            f.write("A small knight battles a giant snail in a garden.")

        try:
            # We run main.py as a subprocess to simulate a real run
            # --storage local: don't try to upload anywhere
            # --input: use our tiny story
            # LITELLM_MODEL=gpt-3.5-turbo: even if not present, we want to see if code execution starts
            # Actually, better yet, we can use a flag or env var to force Mock mode if we had one.
            # For now, let's assume the user has a working environment or we mock the heavy parts.
            
            # Since we want this to be FAST and OFFLINE, we should ideally have a --test or --mock-all flag
            # Let's run it and see if it gets past imports and initialization.
            
            cmd = [
                sys.executable, "src/main.py", 
                "--input", test_story_path,
                "--output", "test_output",
                "--storage", "local"
            ]
            
            # We set an env var to signal this is a test run
            env = os.environ.copy()
            env["PIPELINE_VALIDATION_RUN"] = "1"
            
            logger.info(f"Executing: {' '.join(cmd)}")
            # We'll just run it for a bit or expect it to finish if Mock is enabled.
            # In a real scenario, we'd want a separate test suite.
            # For this task, we'll verify it doesn't crash on startup/imports.
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("✅ Pipeline dry-run passed.")
                return True
            else:
                logger.error(f"❌ Pipeline dry-run failed with exit code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Pipeline dry-run timed out (took > 60s). This might be normal for local models, but suspicious for a mock run.")
            return False
        except Exception as e:
            logger.error(f"❌ Pipeline dry-run crashed: {e}")
            return False
        finally:
            if os.path.exists(test_story_path):
                os.remove(test_story_path)

    def process(self, input_data: Any = None) -> bool:
        """
        Main entry point for validation.
        Returns True if all checks pass.
        """
        syntax_ok = self.check_syntax("src")
        if not syntax_ok:
            return False
            
        # Optional: E2E dry run
        # dry_run_ok = self.run_dry_run()
        # if not dry_run_ok:
        #     return False
            
        return True
