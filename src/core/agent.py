from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the comic generation pipeline.
    """
    def __init__(self, agent_name: str, config: Dict[str, Any] = None):
        self.name = agent_name
        self.config = config or {}
        self.logger = logging.getLogger(f"Agent.{agent_name}")

    def wait_for_user_approval(self, checkpoint_id: str, step_label: str):
        """
        Polls the checkpoint for a 'user_approved' flag for this specific step.
        Used for Human-in-the-Loop orchestration.
        """
        import time
        from src.utils.checkpoint_manager import CheckpointManager
        from src.core.storage import LocalStorage
        
        mgr = CheckpointManager(LocalStorage())
        self.logger.info(f"⏸️ Agent {self.name} is waiting for user approval on step: {step_label}...")
        
        while True:
            state = mgr.load_checkpoint(checkpoint_id)
            if state and state.metadata.get(f"approved_{step_label}"):
                self.logger.info(f"✅ User approved step: {step_label}. Resuming {self.name}...")
                break
            time.sleep(2) # Poll every 2 seconds

    def process(self, input_data: Any) -> Any:
        # To be implemented by subclasses
        pass

    def critique(self, input_data: Any, result: Any) -> Dict[str, Any]:
        """
        Default critique method that passes by default.
        Subclasses can override this to implement self-correction.
        """
        return {"passed": True, "feedback": "No critique implementation; skipping."}

    def validate_output(self, data: Any, expected_schema: Any) -> Any:
        """
        Validates that the data matches the expected Pydantic schema.
        Raises ValidationError if invalid.
        """
        if isinstance(expected_schema, type) and issubclass(expected_schema, BaseModel):
            if isinstance(data, dict):
                return expected_schema.parse_obj(data)
            elif isinstance(data, expected_schema):
                return data
            else:
                 raise ValueError(f"Data type {type(data)} does not match schema {expected_schema}")
        return data

    def run(self, input_data: Any, expected_schema: Any = None) -> Any:
        """
        Orchestrates the process and critique loop with strict validation.
        """
        self.logger.info(f"Starting execution for {self.name}...")
        try:
            # 1. Process
            from src.agents.infrastructure.resilience_agent import safe_retry
            
            @safe_retry(tries=3, delay=1, backoff=2)
            def inner_process():
                return self.process(input_data)
            
            result = inner_process()
            
            # 2. Validation
            if expected_schema:
                try:
                    result = self.validate_output(result, expected_schema)
                except Exception as e:
                    self.logger.error(f"Schema Validation Failed in {self.name}: {e}")
                    raise
            
            # 3. Critique
            critique_result = self.critique(input_data, result)
            if not critique_result.get("passed", True):
                self.logger.warning(f"Critique failed validation: {critique_result.get('feedback')}")
            
            self.logger.info(f"Execution handling complete for {self.name}.")
            return result
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {str(e)}")
            raise

class AgentRegistry:
    """
    Simple registry to keep track of available agents.
    """
    _agents = {}

    @classmethod
    def register(cls, name: str, agent_cls: type[BaseAgent]):
        cls._agents[name] = agent_cls

    @classmethod
    def get(cls, name: str) -> type[BaseAgent]:
        return cls._agents.get(name)
