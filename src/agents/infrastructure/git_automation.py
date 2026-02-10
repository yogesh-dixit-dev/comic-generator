import subprocess
import logging
import os
from typing import Optional
from src.core.agent import BaseAgent

logger = logging.getLogger(__name__)

class GitAutomationAgent(BaseAgent):
    def process(self, commit_message: str = "Auto-update by AI Agent") -> str:
        """
        Stages all changes, commits, and pushes to the remote repository.
        """
        self.logger.info("Starting Git Automation...")
        
        # 1. Check if git is initialized
        if not os.path.isdir(".git"):
            self.logger.error("Not a git repository. Please initialize git first.")
            raise RuntimeError("Current directory is not a git repository.")

        try:
            # 2. Add all changes
            self.logger.info("Staging changes...")
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            
            # 3. Check if there are changes to commit
            status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            if not status.stdout.strip():
                self.logger.info("No changes to commit.")
                return "No changes to commit."

            # 4. Commit
            self.logger.info(f"Committing with message: {commit_message}")
            subprocess.run(["git", "commit", "-m", commit_message], check=True, capture_output=True)
            
            # 5. Push
            self.logger.info("Pushing to remote...")
            result = subprocess.run(["git", "push"], check=True, capture_output=True, text=True)
            
            self.logger.info("Git Push Successful!")
            return f"Success: {result.stdout}"
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Git command failed: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
