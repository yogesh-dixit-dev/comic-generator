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
        Ultra-aggressive JSON extraction.
        Finds the broadest possible {...} or [...] block.
        """
        if not text:
            return "{}"

        # 1. Try markdown block first
        import re
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            candidate = json_match.group(1).strip()
            if candidate: return candidate
            
        # 2. Find outermost curly braces
        start_curly = text.find('{')
        end_curly = text.rfind('}')
        
        # 3. Find outermost square brackets (for list results)
        start_bracket = text.find('[')
        end_bracket = text.rfind(']')
        
        # Determine which one to use
        # Case A: Both found
        if start_curly != -1 and start_bracket != -1:
            if start_curly < start_bracket:
                # Starts with {
                if end_curly > start_curly:
                    return text[start_curly:end_curly+1].strip()
            else:
                # Starts with [
                if end_bracket > start_bracket:
                    return text[start_bracket:end_bracket+1].strip()
        
        # Case B: Only one found
        if start_curly != -1 and end_curly > start_curly:
            return text[start_curly:end_curly+1].strip()
        
        if start_bracket != -1 and end_bracket > start_bracket:
            return text[start_bracket:end_bracket+1].strip()
            
        # 4. Fallback: Clean string and hope for the best
        cleaned = text.strip()
        return cleaned if cleaned else "{}"

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
                
                # Extract schema as JSON
                schema_json = json.dumps(schema.model_json_schema(), indent=2)
                
                # Check if we should override the system prompt for local models
                is_local = "ollama" in self.model_name or "local" in self.model_name
                
                if is_local:
                    # Generate a concrete skeleton template instead of a technical spec
                    def _get_skeleton(json_schema: dict) -> dict:
                        """Recursively creates a minimal skeleton from a JSON schema."""
                        if "properties" in json_schema:
                            obj = {}
                            for k, v in json_schema["properties"].items():
                                if "type" in v:
                                    if v["type"] == "array": obj[k] = []
                                    elif v["type"] == "object": obj[k] = {}
                                    elif v["type"] == "integer": obj[k] = 0
                                    else: obj[k] = "..."
                                else:
                                    obj[k] = "..."
                            return obj
                        return {}

                    try:
                        full_schema = schema.model_json_schema()
                        # We only show the top-level skeleton to keep it concise
                        skeleton = _get_skeleton(full_schema)
                        schema_desc = json.dumps(skeleton, indent=2)
                    except:
                        schema_desc = "{ \"title\": \"...\", \"synopsis\": \"...\", \"scenes\": [] }"

                    if attempt == 0:
                        enhanced_system = (
                            f"{system_prompt}\n\n"
                            f"IMPORTANT: You MUST return ONLY a valid JSON object. No conversation, no preamble.\n"
                            f"CRITICAL: Your output MUST follow this exact top-level structure:\n"
                            f"{schema_desc}\n\n"
                            f"Do NOT invent nested levels (like 'part1' or 'story'). Use the keys above directly."
                        )
                    else:
                        # Even stricter on retries
                        enhanced_system = (
                            f"Your previous output was invalid. You ignored the required structure.\n"
                            f"YOU MUST RETURN ONLY THE DATA matching this structure:\n"
                            f"{schema_desc}\n\n"
                            f"START YOUR RESPONSE WITH '{{' AND END WITH '}}'."
                        )
                else:
                    # For robust models (GPT-4, etc.), use the standard prompt but still include schema if not present
                    if "schema" not in system_prompt.lower():
                        enhanced_system = f"{system_prompt}\n\nSchema: {schema_json}"
                    else:
                        enhanced_system = system_prompt

                response = completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": enhanced_system},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"} if "gpt" in self.model_name or "gemini" in self.model_name else None,
                    api_key=self.api_key,
                    timeout=180 # Longer timeout for Colab
                )
                
                content = response.choices[0].message.content
                if not content or not content.strip():
                    logger.warning(f"Attempt {attempt + 1}: Empty response content.")
                    raise ValueError("LLM returned an empty response")

                # Use ultra-aggressive extractor
                json_content = self._extract_json(content)
                
                # Safety log
                logger.debug(f"Parsing JSON candidate (length {len(json_content)}): {json_content[:100]}...")
                
                if not json_content:
                    raise ValueError("Failed to extract any JSON-like content from response")

                # Repair common mistakes
                import re
                repaired = re.sub(r',\s*([\]}])', r'\1', json_content) # Trailing commas
                
                data = json.loads(repaired)
                return schema.model_validate(data)

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if 'content' in locals() and content:
                    logger.info(f"Raw problematic output: {content[:500]}...")
                
                if attempt < max_retries - 1:
                    import time
                    time.sleep(3 * (attempt + 1))
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
