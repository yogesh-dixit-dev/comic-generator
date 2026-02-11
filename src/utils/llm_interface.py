import os
import threading
from typing import Type, TypeVar, Optional, Any, Dict
from pydantic import BaseModel
import litellm
from litellm import completion
import json
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class LLMInterface:
    _resilience_agent = None  # Class-level cache for efficiency
    _ollama_semaphore = threading.Semaphore(1) # Serialize local LLM calls to prevent timeouts

    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize resilience agent once
        if LLMInterface._resilience_agent is None:
            from src.utils.json_resilience import JSONResilienceAgent
            LLMInterface._resilience_agent = JSONResilienceAgent()

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


    def is_healthy(self) -> bool:
        """
        Checks if the LLM backend is reachable.
        Tries both localhost and 127.0.0.1 for maximum compatibility.
        """
        if "ollama" in self.model_name or "local" in self.model_name:
            import requests
            for host in ["localhost", "127.0.0.1"]:
                try:
                    url = f"http://{host}:11434/api/tags"
                    resp = requests.get(url, timeout=2)
                    if resp.status_code == 200:
                        return True
                except Exception:
                    continue
            return False
        return True # Assume cloud models are healthy if API keys are present

    def generate_structured_output(self, prompt: str, schema: Type[T], system_prompt: str = "You are a helpful assistant. Respond ONLY with valid JSON.") -> T:
        """
        Generates a response from the LLM with optimized latency and resilience.
        """
        max_retries = 3
        last_error = None
        
        # Pre-cache schema skeleton
        resilience = LLMInterface._resilience_agent
        schema_skeleton = resilience.generate_deep_skeleton(schema)
        is_local = "ollama" in self.model_name or "local" in self.model_name
        
        for attempt in range(max_retries):
            # Use semaphore for local models to prevent resource contention
            with LLMInterface._ollama_semaphore if is_local else threading.Lock():
                try:
                    logger.info(f"LLM Request (Attempt {attempt + 1}/{max_retries}) using {self.model_name}...")
                    
                    if is_local:
                        if attempt == 0:
                            enhanced_system = (
                                f"{system_prompt}\n\n"
                                f"IMPORTANT: Respond ONLY with valid JSON.\n"
                                f"CRITICAL: Follow this exact structure:\n"
                                f"{schema_skeleton}\n"
                            )
                        else:
                            enhanced_system = (
                                f"Previous response failed validation. Hallucinated fields detected.\n"
                                f"YOU MUST MATCH THIS EXACT STRUCTURE:\n"
                                f"{schema_skeleton}\n"
                                f"Respond ONLY with the JSON object."
                            )
                    else:
                        enhanced_system = f"{system_prompt}\n\nSchema: {json.dumps(schema.model_json_schema(), indent=2)}"

                    # LiteLLM/Ollama Optimizations
                    completion_kwargs = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": enhanced_system},
                            {"role": "user", "content": prompt}
                        ],
                        "api_key": self.api_key,
                        "timeout": 600
                    }
                    
                    if is_local:
                        completion_kwargs["keep_alive"] = "20m"
                        
                    response = completion(**completion_kwargs)
                    
                    content = response.choices[0].message.content
                    if not content or not content.strip():
                        raise ValueError("Empty response")

                    json_content = self._extract_json(content)
                    
                    if is_local:
                        data = resilience.repair_json(json_content, schema)
                    else:
                        data = json.loads(json_content)

                    return schema.model_validate(data)

                except Exception as e:
                    last_error = e
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    
                    # 💡 Specific logic for connection errors (service might be restarting)
                    is_conn_error = "Connection refused" in str(e) or "APIConnectionError" in str(e)
                    
                    if attempt < max_retries - 1:
                        # Longer sleep for connection errors
                        sleep_time = 1 if is_local and not is_conn_error else 5
                        if is_conn_error:
                             logger.info("⏳ Connection refused. Service might be down or restarting. Waiting 5s...")
                        
                        import time
                        time.sleep(sleep_time)
                    continue
        
        # If we get here, all retries failed
        if last_error:
            raise last_error
        raise ValueError(f"LLM request failed after {max_retries} attempts for unknown reasons.")
        
    def unload_model(self):
        """
        Forcefully unloads the model from Ollama VRAM by setting keep_alive to 0.
        """
        if "ollama" in self.model_name or "local" in self.model_name:
            import requests
            try:
                logger.info(f"📤 Unloading model {self.model_name} from Ollama VRAM...")
                # Ollama API supports 'keep_alive': 0 to unload
                # We use a dummy request to trigger the unload logic
                litellm.completion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "unload"}],
                    keep_alive=0,
                    max_tokens=1
                )
                logger.info("✅ Model unload signal sent.")
            except Exception as e:
                logger.warning(f"⚠️ Failed to unload model: {e}")

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
