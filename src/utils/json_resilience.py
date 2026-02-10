import json
import logging
import re
from typing import Type, Any, Dict, List, TypeVar, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class JSONResilienceAgent:
    """
    Agent specialized in ensuring LLM outputs adhere to specific JSON schemas.
    Handles recursive skeleton generation and heuristic repair of hallucinated fields.
    """
    
    def __init__(self, fast_model_interface=None):
        self.fast_model = fast_model_interface
        # Common Llama hallucinations map
        self.field_aliases = {
            "panel_id": "id",
            "scene_id": "id",
            "location_name": "location",
            "character_name": "name",
            "char_name": "name",
            "visual_description": "description",
            "panel_description": "description",
            "dialog": "dialogue",
            "script_title": "title",
            "story_synopsis": "synopsis"
        }

    def generate_deep_skeleton(self, schema: Type[T]) -> str:
        """
        Generates a recursive, deep JSON skeleton from a Pydantic model.
        Resolves nested models to provide a complete template.
        """
        def _get_default_val(field_type: Any) -> Any:
            # Simple recursive walker using Pydantic's model info
            if hasattr(field_type, "__origin__"): # Handle List, Dict, etc.
                origin = field_type.__origin__
                if origin is list:
                    # Look for the internal type
                    args = field_type.__args__
                    if args and issubclass(args[0], BaseModel):
                        return [_build_obj(args[0])]
                    return ["..."]
                return {}
            
            if issubclass(field_type, BaseModel):
                return _build_obj(field_type)
            
            if field_type is int: return 0
            if field_type is float: return 0.0
            if field_type is bool: return True
            return "..."

        def _build_obj(model: Type[BaseModel]) -> Dict[str, Any]:
            obj = {}
            for name, field in model.model_fields.items():
                # field.annotation gives the type
                obj[name] = _get_default_val(field.annotation)
            return obj

        try:
            skeleton_dict = _build_obj(schema)
            return json.dumps(skeleton_dict, indent=2)
        except Exception as e:
            logger.error(f"Failed to generate deep skeleton: {e}")
            return "{}"

    def repair_json(self, raw_json: str, schema: Type[T]) -> Dict[str, Any]:
        """
        Performs heuristic repair on JSON to align it with the schema.
        Fixes common field hallucinations and structural mismatches.
        """
        try:
            data = json.loads(raw_json)
        except Exception as e:
            # If loads fails, try the ultra-aggressive regex-based repair (pre-loads)
            cleaned = self._pre_process_json_string(raw_json)
            data = json.loads(cleaned)

        # Recursive field mapper
        def _align_fields(current_data: Any, current_schema: Type[BaseModel]):
            if not isinstance(current_data, dict):
                return current_data

            new_data = {}
            target_fields = current_schema.model_fields
            
            for k, v in current_data.items():
                # 1. Direct match
                if k in target_fields:
                    target_key = k
                # 2. Alias match
                elif k.lower() in self.field_aliases:
                    target_key = self.field_aliases[k.lower()]
                    if target_key not in target_fields: target_key = k # Check if target exists in this specific model
                # 3. Case-insensitive match
                else:
                    target_key = k
                    for target in target_fields:
                        if target.lower() == k.lower():
                            target_key = target
                            break
                
                # Recursive step if value is a dict or list of dicts
                field_info = target_fields.get(target_key)
                if field_info:
                    annotation = field_info.annotation
                    # Handle nested model
                    if issubclass(type(annotation), type) and issubclass(annotation, BaseModel):
                        new_data[target_key] = _align_fields(v, annotation)
                    # Handle List[BaseModel]
                    elif hasattr(annotation, "__origin__") and annotation.__origin__ is list:
                        args = annotation.__args__
                        if args and issubclass(args[0], BaseModel):
                            new_data[target_key] = [_align_fields(item, args[0]) for item in v] if isinstance(v, list) else v
                        else:
                            new_data[target_key] = v
                    else:
                        new_data[target_key] = v
                else:
                    # Key not in schema, discard or keep? For now, discard to be "rigorous"
                    pass
            
            return new_data

        return _align_fields(data, schema)

    def _pre_process_json_string(self, text: str) -> str:
        """Fixes trailing commas and unescaped quotes at a string level."""
        # 1. Fix trailing commas before closing braces/brackets
        text = re.sub(r',\s*([\]}])', r'\1', text)
        # 2. Add missing commas between objects in lists (common Llama error)
        text = re.sub(r'}\s*{', '}, {', text)
        text = re.sub(r']\s*\[', '], [', text)
        return text
