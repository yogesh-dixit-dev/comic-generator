import os
import shutil
import zipfile
from abc import ABC, abstractmethod
from typing import Optional, Any, List
import logging

try:
    from huggingface_hub import HfApi, HfFileSystem
except ImportError:
    HfApi = None
    HfFileSystem = None

logger = logging.getLogger(__name__)

class StorageInterface(ABC):
    @abstractmethod
    def save_file(self, local_path: str, remote_path: str) -> str:
        pass
    
    @abstractmethod
    def sync(self, source_dir: str, target_dir: str):
        pass

    @abstractmethod
    def save_comic(self, script: Any, finished_pages: List[str], output_dir: str) -> List[str]:
        """
        Saves the final comic assets and returns a list of output file paths.
        """
        pass

class LocalStorage(StorageInterface):
    """
    Simple local storage. Remote path is just another local path.
    """
    def save_file(self, local_path: str, remote_path: str) -> str:
        # For local, we might just copy, but usually this is a no-op if working in place
        # unless remote_path is a backup dir
        if local_path != remote_path:
             shutil.copy2(local_path, remote_path)
        return remote_path

    def sync(self, source_dir: str, target_dir: str):
        if not os.path.exists(target_dir):
            shutil.copytree(source_dir, target_dir)
        else:
            shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
            
    def save_comic(self, script: Any, finished_pages: List[str], output_dir: str) -> List[str]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_paths = []
        # Save script as JSON for metadata
        script_path = os.path.join(output_dir, "comic_metadata.json")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script.model_dump_json(indent=2))
        output_paths.append(script_path)
        
        # Save pages
        for i, page_path in enumerate(finished_pages):
            if page_path and os.path.exists(page_path):
                ext = os.path.splitext(page_path)[1]
                target_path = os.path.join(output_dir, f"page_{i+1}{ext}")
                shutil.copy2(page_path, target_path)
                output_paths.append(target_path)
                
        return output_paths
            
    def zip_output(self, source_dir: str, zip_name: str = "comic_output.zip") -> str:
        """
        Zips the output directory for easy download.
        """
        zip_path = os.path.join(os.path.dirname(source_dir), zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)
        return zip_path

class HuggingFaceStorage(StorageInterface):
    """
    Storage backend that syncs to a Hugging Face Dataset or Model repo.
    """
    def __init__(self, repo_id: str, token: str, repo_type: str = "dataset"):
        if not HfApi:
            raise ImportError("huggingface_hub library is not installed.")
        self.api = HfApi(token=token)
        self.repo_id = repo_id
        self.repo_type = repo_type
        
        # Ensure repo exists
        try:
            self.api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
            logger.info(f"Connected to Hugging Face Repo: {repo_id}")
        except Exception as e:
            logger.error(f"Failed to create/connect to HF Repo: {e}")
            raise

    def save_file(self, local_path: str, remote_path: str) -> str:
        """
        Uploads a single file.
        """
        self.api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=self.repo_id,
            repo_type=self.repo_type
        )
        return f"hf://{self.repo_id}/{remote_path}"

    def sync(self, source_dir: str, target_dir: str = ""):
        """
        Uploads an entire directory.
        """
        logger.info(f"Uploading {source_dir} to Hugging Face {self.repo_id}...")
        self.api.upload_folder(
            folder_path=source_dir,
            path_in_repo=self._normalize_path(target_dir),
            repo_id=self.repo_id,
            repo_type=self.repo_type
        )

    def save_comic(self, script: Any, finished_pages: List[str], output_dir: str) -> List[str]:
        """
        Saves comic locally first, then syncs to HF.
        """
        local_storage = LocalStorage()
        output_paths = local_storage.save_comic(script, finished_pages, output_dir)
        
        # Sync the entire folder to HF
        self.sync(output_dir, "comic_output")
        
        return [f"hf://{self.repo_id}/comic_output/{os.path.basename(p)}" for p in output_paths]

    def _normalize_path(self, path: str) -> str:
        """Converts Windows paths to HF-compatible forward slash paths."""
        return path.replace("\\", "/")
