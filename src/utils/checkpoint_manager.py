import os
import json
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
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
        self._executor = ThreadPoolExecutor(max_workers=2) # Background sync/progress tasks
        
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

        # 2. Background Cloud Sync & Progress Update
        if isinstance(self.storage, HuggingFaceStorage):
            # Normalization and sync logic offloaded to background
            self._executor.submit(self._background_sync, path, state)
        else:
            # Local/GitHub sync can also be backgrounded
            self._executor.submit(self._background_git_push, path, state.input_hash)

    def _background_sync(self, path: str, state: PipelineState):
        """Internal method for background HF sync."""
        try:
            remote_path = path.replace("\\", "/")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.storage.save_file(path, remote_path)
                    logger.info(f"☁️ Background checkpoint sync complete (Attempt {attempt+1}).")
                    break
                except Exception as hf_err:
                    if attempt == max_retries - 1:
                        raise hf_err
                    logger.warning(f"HF background upload attempt {attempt+1} failed: {hf_err}. Retrying...")
            
            self.update_live_progress(state)
        except Exception as e:
            logger.warning(f"⚠️ Background HF sync failed: {e}")

    def _background_git_push(self, path: str, input_hash: str):
        """Internal method for background Git sync."""
        try:
            from src.agents.infrastructure.git_automation import GitAutomationAgent
            git = GitAutomationAgent("CheckpointGit")
            
            # Ensure git identity is set (crucial for Colab/CI)
            self._ensure_git_config(git)
            
            git.run_command(f"git add {path}")
            git.run_command(f'git commit -m "docs: save checkpoint for {input_hash[:12]}"')
            git.run_command("git push")
            logger.info("☁️ Background git sync complete.")
        except Exception as e:
            logger.debug(f"Git background sync skipped or failed: {e}")

    def _ensure_git_config(self, git_agent):
        """Sets a dummy git identity if none exists."""
        try:
            # Check if user.email is set
            check = git_agent.run_command("git config user.email")
            if not check or "root@" in check or "(none)" in check:
                logger.info("🔧 Setting dummy Git identity for checkpoint sync...")
                git_agent.run_command('git config --global user.email "comic-gen-bot@example.com"')
                git_agent.run_command('git config --global user.name "ComicGen Bot"')
        except Exception:
            # Fallback for systems where git config fails
            pass

    def shutdown(self, wait: bool = True):
        """Shuts down the background executor."""
        self._executor.shutdown(wait=wait)
        logger.info("⚙️ CheckpointManager background executor shut down.")

    def update_live_progress(self, state: PipelineState):
        """
        Writes a lightweight progress status to the root of the cloud storage.
        Allows users to monitor status without downloading heavy checkpoints.
        """
        progress_data = {
            "input_hash": state.input_hash,
            "stage": state.stage,
            "last_chunk": state.last_chunk_index,
            "scenes_completed": state.last_scene_id,
            "pages_generated": len(state.finished_pages),
            "timestamp": str(os.path.getmtime(self.get_checkpoint_path(state.input_hash))) if os.path.exists(self.get_checkpoint_path(state.input_hash)) else "N/A"
        }
        
        local_progress_path = os.path.join(self.checkpoint_dir, "progress.json")
        with open(local_progress_path, "w") as f:
            json.dump(progress_data, f, indent=2)
            
        if isinstance(self.storage, HuggingFaceStorage):
            try:
                self.storage.save_file(local_progress_path, "progress.json")
                logger.info("📊 Live progress updated on Hugging Face.")
            except Exception as e:
                logger.warning(f"Failed to update live progress: {e}")

    def load_checkpoint(self, input_hash: str) -> Optional[PipelineState]:
        """Loads the checkpoint for a specific input hash, pulling from cloud if needed."""
        path = self.get_checkpoint_path(input_hash)
        
        # 1. Try cloud pull if local missing
        if not os.path.exists(path) and isinstance(self.storage, HuggingFaceStorage):
            try:
                logger.info(f"🔍 Checkpoint missing locally. Attempting to pull from Hugging Face: {path}")
                # Use HuggingFace Hub API to download single file if it exists
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id=self.storage.repo_id,
                    filename=path.replace("\\", "/"), # HF uses forward slashes
                    repo_type="dataset",
                    local_dir=".",
                    token=self.storage.api.token
                )
                logger.info("☁️ Successfully pulled checkpoint from Hugging Face.")
            except Exception as e:
                logger.debug(f"Hugging Face pull failed or file does not exist: {e}")

        if not os.path.exists(path):
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Safe Unmarshalling: Try strict first, then fallback to dict-level repair
            state = PipelineState.model_validate(data)
            logger.info(f"🔄 Checkpoint loaded successfully from {path}")
            return state
            
        except ValidationError as ve:
            logger.warning(f"⚠️ Checkpoint schema mismatch: {ve}. Attempting partial recovery...")
            # Fallback: try to return the raw data if we can't validate (for development)
            try: return PipelineState.model_construct(**data)
            except: return None
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return None

    def clear_checkpoint(self, input_hash: str):
        """Deletes the checkpoint file."""
        path = self.get_checkpoint_path(input_hash)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"🗑️ Checkpoint {path} cleared.")

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Lists all available checkpoints with summary info."""
        checkpoints = []
        if not os.path.exists(self.checkpoint_dir):
            return []
            
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith("checkpoint_") and filename.endswith(".json"):
                # Extract hash if possible, or just load and check
                path = os.path.join(self.checkpoint_dir, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # We only need a summary for the list view
                        checkpoints.append({
                            "input_hash": data.get("input_hash"),
                            "stage": data.get("stage", "unknown"),
                            "last_chunk_index": data.get("last_chunk_index", -1),
                            "scenes_total": len(data.get("master_script", {}).get("scenes", [])) if data.get("master_script") else 0,
                            "pages_generated": len(data.get("finished_pages", [])),
                            "timestamp": os.path.getmtime(path),
                            "name": data.get("metadata", {}).get("project_name", f"Project {data.get('input_hash', '')[:8]}")
                        })
                except Exception as e:
                    logger.warning(f"Failed to parse checkpoint {filename}: {e}")
        
        # Sort by timestamp descending
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints
