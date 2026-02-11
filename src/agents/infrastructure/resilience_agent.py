import logging
import time
import functools
import os
from typing import Any, Dict, List, Optional, Callable, Type, Tuple
from src.core.agent import BaseAgent

logger = logging.getLogger(__name__)

class ResilienceAgent(BaseAgent):
    """
    Agent responsible for making the repository and pipeline failure-proof.
    Provides retry logic, health checks, and fallback mechanisms.
    """
    def __init__(self, agent_name: str = "ResilienceAgent", config: Dict[str, Any] = None):
        super().__init__(agent_name, config)
        self.max_retries = self.config.get("max_retries", 3)
        self.backoff_factor = self.config.get("backoff_factor", 2)
        self.model_hierarchy = [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "ollama/llama3",
            "mock"
        ]

    def retry(self, exceptions: Tuple[Type[Exception], ...] = (Exception,), 
              tries: int = 3, delay: int = 1, backoff: int = 2):
        """
        Retry decorator with exponential backoff.
        """
        def decorator_retry(func: Callable):
            @functools.wraps(func)
            def wrapper_retry(*args, **kwargs):
                mtries, mdelay = tries, delay
                while mtries > 1:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        msg = f"{str(e)}, Retrying in {mdelay} seconds..."
                        self.logger.warning(msg)
                        time.sleep(mdelay)
                        mtries -= 1
                        mdelay *= backoff
                return func(*args, **kwargs)
            return wrapper_retry
        return decorator_retry

    def check_system_health(self) -> Dict[str, Any]:
        """
        Performs a comprehensive health check of the environment.
        """
        health = {
            "status": "healthy",
            "checks": {}
        }
        
        # 1. Check disk space
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (2**30)
        health["checks"]["disk_space"] = f"{free_gb} GB free"
        if free_gb < 1:
            health["status"] = "degraded"
            health["checks"]["disk_space"] += " (CRITICAL: Low Space)"

        # 2. Check environment variables
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [v for v in required_vars if not os.getenv(v)]
        health["checks"]["env_vars"] = "OK" if not missing_vars else f"Missing: {', '.join(missing_vars)}"
        if missing_vars:
             health["status"] = "degraded"

        # 3. Check LLM Local Backend
        import requests
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            health["checks"]["local_llm"] = "Ollama Online" if resp.status_code == 200 else "Ollama Offline"
        except:
            health["checks"]["local_llm"] = "Ollama Not Reachable"

        self.logger.info(f"System Health Check Results: {health}")
        return health

    def get_fallback_model(self, failing_model: str) -> str:
        """
        Returns the next best model in the hierarchy if one fails.
        """
        try:
            current_index = self.model_hierarchy.index(failing_model)
            if current_index + 1 < len(self.model_hierarchy):
                fallback = self.model_hierarchy[current_index + 1]
                self.logger.info(f"Fallback triggered: {failing_model} -> {fallback}")
                return fallback
        except ValueError:
            pass
        
        return "mock" # Global fallback

    def process(self, input_data: Any = None) -> Dict[str, Any]:
        """Main entry point for health reporting."""
        return self.check_system_health()

# Singleton-equivalent instance for easy decorator usage
_resilience_instance = None

def get_resilience_agent() -> ResilienceAgent:
    global _resilience_instance
    if _resilience_instance is None:
        _resilience_instance = ResilienceAgent()
    return _resilience_instance

def safe_retry(tries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """Simple wrapper for using the resilience agent's retry as a decorator."""
    return get_resilience_agent().retry(tries=tries, delay=delay, backoff=backoff, exceptions=exceptions)
