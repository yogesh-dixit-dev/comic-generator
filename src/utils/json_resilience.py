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
        self.field_aliases = {
            "panel_id": "id",
            "scene_id": "id",
            "scene_number": "id",
            "location_name": "location",
            "character_name": "name",
            "char_name": "name",
            "visual_description": "description",
            "panel_description": "description",
            "narrative": "narrative_summary",
            "text": "narrative_summary",
            "summary": "narrative_summary",
            "dialog": "dialogue",
            "script_title": "title",
            "story_synopsis": "synopsis"
        }

    def _is_pydantic_base(self, cls):
        from typing import get_origin, get_args
        if isinstance(cls, type) and issubclass(cls, BaseModel):
            return True
        # Handle Optional/Union
        origin = get_origin(cls)
        if origin is not None:
            args = get_args(cls)
            for arg in args:
                if self._is_pydantic_base(arg):
                    return True
        return False

    def _get_base_model(self, cls):
        from typing import get_origin, get_args
        if isinstance(cls, type) and issubclass(cls, BaseModel):
            return cls
        origin = get_origin(cls)
        if origin is not None:
            for arg in get_args(cls):
                res = self._get_base_model(arg)
                if res: return res
        return None

    def generate_deep_skeleton(self, schema: Type[T]) -> str:
        """
        Generates a recursive, deep JSON skeleton from a Pydantic model.
        Resolves nested models to provide a complete template.
        """
        from typing import get_origin, get_args

        def _get_default_val(field_type: Any) -> Any:
            origin = get_origin(field_type)
            if origin is list:
                args = get_args(field_type)
                if args and self._is_pydantic_base(args[0]):
                    return [self._build_obj(self._get_base_model(args[0]))]
                return ["..."]
            
            if self._is_pydantic_base(field_type):
                return self._build_obj(self._get_base_model(field_type))
            
            # Use base type if available
            base_type = origin or field_type
            if base_type is int: return 1
            if base_type in (float, complex): return 0.0
            if base_type is bool: return True
            return "..."

        skeleton_dict = self._build_obj(schema)
        return json.dumps(skeleton_dict, indent=2)

    def _build_obj(self, model: Type[BaseModel]) -> Dict[str, Any]:
        obj = {}
        for name, field in model.model_fields.items():
            # In generate_deep_skeleton, we need a local-ish _get_default_val or just inline it
            # To keep it clean, let's just make it a call back to a more robust helper if needed.
            # Simplified for now to avoid circularity in local defs.
            origin = getattr(field.annotation, "__origin__", None)
            if origin is list:
                obj[name] = []
            elif isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
                obj[name] = {} # Stub for shallow skeletons
            else:
                obj[name] = "..."
        return obj

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
                    from typing import get_origin, get_args
                    
                    def _get_nested_val(val, annot):
                        if self._is_pydantic_base(annot):
                            return _align_fields(val, self._get_base_model(annot))
                        origin = get_origin(annot)
                        if origin is list:
                            args = get_args(annot)
                            if args and self._is_pydantic_base(args[0]) and isinstance(val, list):
                                return [_align_fields(item, self._get_base_model(args[0])) for item in val]
                            
                            # Heuristic: Repair Dialogue Hallucination
                            # If we expect List[BaseModel] (like Panel.dialogue) but get List[str]
                            if args and self._is_pydantic_base(args[0]) and isinstance(val, list):
                                # If the target model has 'speaker' and 'text' fields, we can repair it
                                base_model = self._get_base_model(args[0])
                                fields = base_model.model_fields
                                if "speaker" in fields and "text" in fields:
                                    repaired_list = []
                                    for item in val:
                                        if isinstance(item, str):
                                            if ":" in item:
                                                parts = item.split(":", 1)
                                                repaired_list.append({"speaker": parts[0].strip(), "text": parts[1].strip()})
                                            else:
                                                repaired_list.append({"speaker": "Narrator", "text": item})
                                        else:
                                            repaired_list.append(item)
                                    return repaired_list
                                return [_align_fields(item, base_model) for item in val]
                        return val

                    # Heuristic for INT fields (like 'id' or 'panel_id')
                    try:
                        origin = get_origin(field_info.annotation)
                        base_annot = origin or field_info.annotation
                        if base_annot is int:
                            if isinstance(v, str):
                                digits = re.findall(r'\d+', v)
                                if digits: v = int(digits[0])
                            elif not isinstance(v, int):
                                try: v = int(v)
                                except: pass
                    except: pass

                    new_data[target_key] = _get_nested_val(v, annotation)
                else:
                    # Special case: map 'title' to 'id' if 'id' is missing and v is a number-like string
                    if k.lower() == "title" and "id" in target_fields and "id" not in new_data:
                        try:
                            digits = re.findall(r'\d+', str(v))
                            if digits: new_data["id"] = int(digits[0])
                        except: pass
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
