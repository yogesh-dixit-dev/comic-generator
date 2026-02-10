from abc import ABC, abstractmethod
from typing import Optional

class ImageGeneratorInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024, seed: Optional[int] = None) -> str:
        """
        Generates an image from a prompt.
        Returns the path to the saved image file.
        """
        pass
