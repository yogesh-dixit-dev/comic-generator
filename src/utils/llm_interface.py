import os
from typing import Type, TypeVar, Optional, Any, Dict
from pydantic import BaseModel
import litellm
from litellm import completion
import json
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class LLMInterface:
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") # Fallback to env var
        # litellm handles API keys from env vars automatically for many providers

    def generate_structured_output(self, prompt: str, schema: Type[T], system_prompt: str = "You are a helpful assistant.") -> T:
        """
        Generates a response from the LLM that strictly enforces the given Pydantic schema.
        """
        try:
            logger.info(f"Sending request to LLM ({self.model_name})...")
            
            # Use LiteLLM's function calling or JSON mode to enforce schema
            # For simplicity with variable providers, we'll try to use the 'response_format' if supported (OpenAI)
            # or standard prompting with Pydantic schema dump.
            
            response = completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"} if "gpt" in self.model_name else None,
                api_key=self.api_key
            )
            
            content = response.choices[0].message.content
            logger.debug(f"LLM Response: {content}")
            
            # Naive parsing for now - robust implementation would handle retries / json repair
            data = json.loads(content)
            return schema.model_validate(data)

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            # Fallback strategy could be added here
            raise

    def generate_text(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Standard text generation.
        """
        response = completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            api_key=self.api_key
        )
        return response.choices[0].message.content
