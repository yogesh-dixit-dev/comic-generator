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
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            text = text[start_index:end_index+1].strip()
        
        # 3. Basic JSON Repair (handle trailing commas)
        # Remove trailing commas before closing braces/brackets
        text = re.sub(r',\s*([\]}])', r'\1', text)
        
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
                enhanced_system = f"{system_prompt}\n\nYou MUST return a JSON object that matches this EXACT schema:\n{schema_json}"
            else:
                enhanced_system = system_prompt

            response = completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system},
                    {"role": "user", "content": prompt}
                ],
                # response_format is unreliable mainly for OpenAI. 
                # For Ollama/Gemini, we rely on strict prompting and our robust _extract_json.
                response_format={"type": "json_object"} if "gpt" in self.model_name else None,
                api_key=self.api_key
            )
            
            content = response.choices[0].message.content
            if not content:
                logger.error("LLM returned an empty response!")
                raise ValueError("Empty response from LLM")

            logger.info(f"Raw LLM Response (first 100 chars): {content[:100]}...")
            
            # Use robust extractor
            json_content = self._extract_json(content)
            
            try:
                data = json.loads(json_content)
                logger.debug(f"Successfully parsed keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            except json.JSONDecodeError as je:
                logger.error(f"JSON Decode Error: {je}")
                logger.error(f"Full problematic content: \n{content}")
                # Try to return a minimal valid structure to see if it allows the process to continue
                # But for a ScriptWriter, we need real data.
                raise

            return schema.model_validate(data)

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            if 'content' in locals():
                 logger.error(f"Full content that failed: {content}")
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
