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
        Extracts and cleans JSON content from a string.
        Handles markdown blocks, conversational preamble, and common LLM mistakes.
        """
        if not text or not text.strip():
            return "{}"

        # Trim whitespace
        text = text.strip()
        
        # 1. Check for markdown code blocks
        import re
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            text = json_match.group(1).strip()
            
        # 2. Find the outermost '{' and '}'
        import re
        # Look for first '{' that likely starts a key, or just any '{'
        start_match = re.search(r'\{\s*"', text)
        start_index = start_match.start() if start_match else text.find('{')
        end_index = text.rfind('}')
        
        if start_index != -1 and end_index != -1 and end_index >= start_index:
            text = text[start_index:end_index+1].strip()
        
        # 3. Basic JSON Repair (handle trailing commas)
        # Remove trailing commas before closing braces/brackets
        text = re.sub(r',\s*([\]}])', r'\1', text)
        
        # Final safety check: if we somehow got an empty string, return empty object
        return text or "{}"

    def generate_structured_output(self, prompt: str, schema: Type[T], system_prompt: str = "You are a helpful assistant. Respond ONLY with valid JSON.") -> T:
        """
        Generates a response from the LLM that strictly enforces the given Pydantic schema.
        Includes a retry mechanism for robustness.
        """
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"LLM Request (Attempt {attempt + 1}/{max_retries}) using {self.model_name}...")
                
                # Augment system prompt for smaller models IF not already detailed
                if ("ollama" in self.model_name or "gemini" in self.model_name) and "schema" not in system_prompt.lower():
                    schema_json = json.dumps(schema.model_json_schema(), indent=2)
                    # For retries, we might want an even simpler prompt
                    if attempt > 0:
                        enhanced_system = f"Respond ONLY with a JSON object matching this schema: {schema_json}. No talk."
                    else:
                        enhanced_system = f"{system_prompt}\n\nYou MUST return a JSON object that matches this EXACT schema:\n{schema_json}"
                else:
                    enhanced_system = system_prompt

                response = completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": enhanced_system},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"} if "gpt" in self.model_name else None,
                    api_key=self.api_key,
                    timeout=120 # Give local models more time
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("LLM returned an empty response")

                logger.debug(f"Raw Response: {content[:200]}...")
                
                # Use robust extractor
                json_content = self._extract_json(content)
                data = json.loads(json_content)
                
                return schema.model_validate(data)

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 * (attempt + 1)) # Exponential backoff
                continue
        
        logger.error(f"All {max_retries} attempts failed for LLM generation.")
        raise last_error

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
