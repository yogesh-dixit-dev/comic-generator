import os
import json
import hashlib
import logging
from typing import Optional, Any, Dict
from pydantic import ValidationError
from src.core.checkpoint import PipelineState
from src.core.storage import StorageInterface, LocalStorage, HuggingFaceStorage

logger = logging.getLogger("CheckpointManager")

class CheckpointManager:
    """
    Manages saving and loading of pipeline states with cloud persistence.
    """
    def __init__(self, storage: StorageInterface, checkpoint_dir: str = ".checkpoints"):
        self.storage = storage
        self.checkpoint_dir = checkpoint_dir
        if isinstance(storage, LocalStorage):
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

    def get_input_hash(self, file_path: str) -> str:
        """Generates a Blake2b hash of the input file content."""
        if not os.path.exists(file_path):
            return "unknown"
        
        hasher = hashlib.blake2b()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_checkpoint_path(self, input_hash: str) -> str:
        """Returns the relative path for the checkpoint file."""
        return os.path.join(self.checkpoint_dir, f"checkpoint_{input_hash[:12]}.json")

    def save_checkpoint(self, state: PipelineState):
        """Saves the state locally and attempts to sync to cloud."""
        path = self.get_checkpoint_path(state.input_hash)
        
        # 1. Save Locally
        with open(path, 'w') as f:
            f.write(state.model_dump_json(indent=2))
        
        logger.info(f"💾 Checkpoint saved locally to {path}")

        # 2. Cloud Sync
        try:
            if isinstance(self.storage, HuggingFaceStorage):
                self.storage.save_file(path, path)
                logger.info("☁️ Checkpoint synced to Hugging Face.")
            else:
                # If local, we rely on the user running git push or a GitAutomationAgent call
                # In Colab/dev mode, we might want to trigger a git commit for the checkpoint specifically
                from src.agents.infrastructure.git_automation import GitAutomationAgent
                git = GitAutomationAgent("CheckpointGit")
                git.run_command(f"git add {path}")
                git.run_command(f'git commit -m "docs: save checkpoint for {state.input_hash[:12]}"')
                git.run_command("git push")
                logger.info("☁️ Checkpoint pushed to GitHub.")
        except Exception as e:
            logger.warning(f"⚠️ Cloud sync for checkpoint failed: {e}. Progress is still saved locally.")

    def load_checkpoint(self, input_hash: str) -> Optional[PipelineState]:
        """Loads the checkpoint for a specific input hash."""
        path = self.get_checkpoint_path(input_hash)
        
        if not os.path.exists(path):
            return None

        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Safe Unmarshalling: Try strict first, then fallback to dict-level repair
            state = PipelineState.model_validate(data)
            logger.info(f"🔄 Checkpoint loaded successfully from {path}")
            return state
            
        except ValidationError as ve:
            logger.warning(f"⚠️ Checkpoint schema mismatch: {ve}. Attempting partial recovery...")
            # Fallback: We might have updated models. We should try to extract what we can.
            # For now, we return None to be safe, or we could implement a migration.
            return None
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return None

    def clear_checkpoint(self, input_hash: str):
        """Deletes the checkpoint file."""
        path = self.get_checkpoint_path(input_hash)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"🗑️ Checkpoint {path} cleared.")
