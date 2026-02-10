import os
import shutil
import zipfile
from abc import ABC, abstractmethod
from typing import Optional
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
            path_in_repo=target_dir,
            repo_id=self.repo_id,
            repo_type=self.repo_type
        )
