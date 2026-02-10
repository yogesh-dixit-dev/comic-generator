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
            "setting": "location",
            "character_name": "name",
            "char_name": "name",
            "visual_description": "description",
            "panel_description": "description",
            "image": "description",
            "image_description": "description",
            "appearance": "description",
            "narrative": "narrative_summary",
            "text": "narrative_summary",
            "summary": "narrative_summary",
            "scene": "narrative_summary",
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
                            # If we expect List[BaseModel] or List[Dict] (like Panel.dialogue) but get List[str]
                            is_dict = False
                            try: is_dict = issubclass(args[0], dict)
                            except: pass

                            if args and (self._is_pydantic_base(args[0]) or is_dict) and isinstance(val, list):
                                # If the target looks like a dialogue item (via dict keys or model fields)
                                target_fields = []
                                if is_dict:
                                    target_fields = ["speaker", "text"]
                                else:
                                    target_fields = self._get_base_model(args[0]).model_fields
                                
                                if "speaker" in target_fields and "text" in target_fields:
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
                                
                                if not is_dict:
                                    return [_align_fields(item, self._get_base_model(args[0])) for item in val]
                        return val

                    # Heuristic for INT/FLOAT fields
                    try:
                        origin = get_origin(field_info.annotation)
                        base_annot = origin or field_info.annotation
                        
                        if base_annot in (int, float):
                            current_val = v
                            if isinstance(v, str):
                                # 1. Handle fractions like "6/10"
                                if "/" in v:
                                    try:
                                        num, den = v.split("/")
                                        current_val = float(num) / float(den) * 10.0 # Standardize to 10
                                    except: pass
                                else:
                                    # 2. Extract first number
                                    digits = re.findall(r'[-+]?\d*\.\d+|\d+', v)
                                    if digits: current_val = float(digits[0])
                            
                            # 3. Handle normalization (if score > 10, assume /100)
                            if isinstance(current_val, (int, float)) and current_val > 10 and "score" in target_key.lower():
                                current_val = current_val / 10.0
                            
                            # 4. Final Type cast
                            if base_annot is int:
                                v = int(float(current_val))
                            else:
                                v = float(current_val)
                    except: pass

                    # Heuristic for BOOL fields
                    try:
                        origin = get_origin(field_info.annotation)
                        base_annot = origin or field_info.annotation
                        if base_annot is bool:
                            if isinstance(v, str):
                                if v.lower() in ("yes", "true", "y", "1", "pass"): v = True
                                elif v.lower() in ("no", "false", "n", "0", "fail"): v = False
                    except: pass

                    new_data[target_key] = _get_nested_val(v, annotation)
                else:
                    # Key not in schema, keep it for fallback processing
                    pass
            
            # Post-alignment fallbacks for required fields
            # 1. Sequence ID Generation
            if "id" in target_fields and "id" not in new_data:
                # If we are in recursive mode we don't know the index, but we can try to find an excess field that looks like un ID
                # or just set a placeholder. 
                # Better yet: the parent list processor should handle this.
                pass

            # 2. Scene-Specific Fallbacks
            if "location" in target_fields and not new_data.get("location"):
                for fallback in ["setting", "title", "description", "scene"]:
                    if current_data.get(fallback) and len(str(current_data.get(fallback))) < 100:
                        new_data["location"] = str(current_data.get(fallback))
                        break
                if not new_data.get("location"):
                    new_data["location"] = "Unknown Location"

            if "narrative_summary" in target_fields and not new_data.get("narrative_summary"):
                for fallback in ["scene", "description", "text", "summary"]:
                    if current_data.get(fallback):
                        new_data["narrative_summary"] = str(current_data.get(fallback))
                        break

            # 3. Personality Fallbacks
            if "personality" in target_fields and not new_data.get("personality"):
                # If we have description but no personality, or if personality is just missing
                new_data["personality"] = "Determined Character"

            # 4. Style Guide Dict-to-String
            if "style_guide" in target_fields and isinstance(current_data.get("style_guide"), dict):
                style_dict = current_data.get("style_guide")
                style_str = "; ".join([f"{k}: {v}" for k, v in style_dict.items()])
                new_data["style_guide"] = style_str

            # 5. Critique Result Dict-to-Bool
            if "passed" in target_fields and isinstance(current_data.get("passed"), dict):
                # If passed is a dict like {"visuals": True, ...}, we take the min
                new_data["passed"] = all(current_data.get("passed").values())

            return new_data

        # Wrap _align_fields to handle list sequence IDs
        def _process_item(item, schema_type, index):
            aligned = _align_fields(item, schema_type)
            if "id" in schema_type.model_fields and (aligned.get("id") is None):
                aligned["id"] = index + 1
            return aligned

        # Top level return with index awareness for lists
        if isinstance(data, list) and self._is_pydantic_base(schema):
            return [_process_item(item, self._get_base_model(schema), i) for i, item in enumerate(data)]
        
        # If it's a dict
        final_data = _align_fields(data, schema)
        
        # Ensure top level lists (like scenes) have IDs
        if "scenes" in final_data and isinstance(final_data["scenes"], list):
            from src.core.models import Scene
            final_data["scenes"] = [_process_item(s, Scene, i) for i, s in enumerate(final_data["scenes"])]
            
        return final_data

    def _pre_process_json_string(self, text: str) -> str:
        """Fixes common structural JSON errors at a raw string level."""
        # 0. Strip leading/trailing non-JSON noise (markdown fences etc)
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
            
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # 1. Fix trailing commas before closing braces/brackets
        text = re.sub(r',\s*([\]}])', r'\1', text)
        
        # 2. Add missing commas between objects in lists (common Llama error)
        text = re.sub(r'}\s*{', '}, {', text)
        text = re.sub(r']\s*\[', '], [', text)
        
        # 3. Add missing commas between key-value pairs (aggressive)
        # Matches: "key": value "key":
        # We look for a pattern where a value (string, number, or boolean) is followed by a quote without a comma.
        # This is strictly a fallback for when the initial parse fails.
        text = re.sub(r'("\s*:\s*(?:(?:"[^"]*")|(?:\d+(?:\.\d+)?)|(?:true|false|null)))\s*(")', r'\1, \2', text)
        
        # 4. Handle truncated JSON (unclosed brackets/braces)
        # If the string ends abruptly, try to close it. 
        # (This is very heuristic, but better than a hard crash)
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        if open_braces > 0: text += '}' * open_braces
        if open_brackets > 0: text += ']' * open_brackets
        
        return text
