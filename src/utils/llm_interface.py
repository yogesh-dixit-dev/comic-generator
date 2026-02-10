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

    def _extract_json(self, text: str) -> str:
        """
        Extracts JSON content from a string that might be wrapped in markdown code blocks
        or contains conversational preamble.
        """
        # Trim whitespace
        text = text.strip()
        
        # 1. Check for markdown code blocks
        import re
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            return json_match.group(1).strip()
            
        # 2. Find the outermost '{' and '}'
        # We look for the first '{' that likely starts a key
        import re
        start_match = re.search(r'\{\s*"', text)
        start_index = start_match.start() if start_match else text.find('{')
        end_index = text.rfind('}')
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return text[start_index:end_index+1].strip()
            
        return text

    def generate_structured_output(self, prompt: str, schema: Type[T], system_prompt: str = "You are a helpful assistant. Respond ONLY with valid JSON.") -> T:
        """
        Generates a response from the LLM that strictly enforces the given Pydantic schema.
        """
        try:
            logger.info(f"Sending request to LLM ({self.model_name})...")
            
            # Augment system prompt for smaller models IF not already detailed
            if ("ollama" in self.model_name or "gemini" in self.model_name) and "schema" not in system_prompt.lower():
                schema_json = json.dumps(schema.model_json_schema(), indent=2)
                enhanced_system = f"{system_prompt}\n\nStrictly follow this JSON schema:\n{schema_json}"
            else:
                enhanced_system = system_prompt

            response = completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system},
                    {"role": "user", "content": prompt}
                ],
                # response_format is reliable mainly for OpenAI/Grok. 
                # For Ollama/Gemini, we rely on strict prompting and our robust _extract_json.
                response_format={"type": "json_object"} if "gpt" in self.model_name else None,
                api_key=self.api_key
            )
            
            content = response.choices[0].message.content
            logger.info(f"Raw LLM Response (first 200 chars): {content[:200]}...")
            
            # Use robust extractor
            json_content = self._extract_json(content)
            logger.debug(f"Extracted JSON Content: {json_content}")
            
            data = json.loads(json_content)
            logger.info(f"Parsed JSON keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            return schema.model_validate(data)

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            if 'data' in locals():
                logger.error(f"Data that failed validation: {data}")
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
