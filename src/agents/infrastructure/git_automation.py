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
        
        # 0. Core Quality Check
        from src.agents.infrastructure.qa_agent import QualityAssuranceAgent
        qa = QualityAssuranceAgent()
        if not qa.run_all_checks():
            self.logger.error("Abort: Quality checks failed. Fix the issues before pushing.")
            return "Failure: Quality checks failed."

        # 1. Check if git is initialized
        if not os.path.isdir(".git"):
            self.logger.error("Not a git repository. Please initialize git first.")
            raise RuntimeError("Current directory is not a git repository.")

        try:
            # 2. Add all changes
            self.logger.info("Staging changes...")
            self.run_command("git add .")
            
            # 3. Check if there are changes to commit
            status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            if not status.stdout.strip():
                self.logger.info("No changes to commit.")
                return "No changes to commit."

            # 4. Commit
            self.logger.info(f"Committing with message: {commit_message}")
            self.run_command(f'git commit -m "{commit_message}"')
            
            # 5. Push
            self.logger.info("Pushing to remote...")
            self.run_command("git push")
            
            self.logger.info("Git Push Successful!")
            return "Success: Changes pushed to remote."
            
        except Exception as e:
            error_msg = f"Git automation failed: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def run_command(self, command: str) -> str:
        """Runs a raw git command and returns output."""
        try:
            # We use shell=True carefully for git commands
            import shlex
            args = shlex.split(command)
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Git command '{command}' failed: {e.stderr}")
            raise RuntimeError(e.stderr)
