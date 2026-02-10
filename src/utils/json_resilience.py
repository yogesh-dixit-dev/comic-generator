import json
import logging
import re
from typing import Type, Any, Dict, List, TypeVar, Optional, get_origin, get_args
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
            "dialogue": "dialogue",
            "script_title": "title",
            "story_synopsis": "synopsis",
            "goal": "personality",
            "goals": "personality",
            "motivation": "personality",
            "motivations": "personality",
            "personality_traits": "personality",
            "traits": "personality",
            "background": "personality",
            "occupation": "personality",
            "notes": "personality"
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
        skeleton_dict = self._build_obj(schema)
        return json.dumps(skeleton_dict, indent=2)

    def _build_obj(self, model: Type[BaseModel]) -> Dict[str, Any]:
        """Recursively builds a template object for a Pydantic model."""
        obj = {}
        target_fields = model.model_fields
        
        for name, field in target_fields.items():
            field_type = field.annotation
            
            # 1. Handle List Types
            origin = get_origin(field_type)
            if origin is list:
                inner_args = get_args(field_type)
                if inner_args and self._is_pydantic_base(inner_args[0]):
                    obj[name] = [self._build_obj(self._get_base_model(inner_args[0]))]
                else:
                    obj[name] = ["..."]
                continue
            
            # 2. Handle Nested Pydantic Models
            if self._is_pydantic_base(field_type):
                obj[name] = self._build_obj(self._get_base_model(field_type))
                continue
            
            # 3. Handle Primitive Fallbacks
            base_type = origin or field_type
            if base_type is int: obj[name] = 1
            elif base_type in (float, complex): obj[name] = 0.0
            elif base_type is bool: obj[name] = True
            elif base_type is str: obj[name] = "..."
            else: obj[name] = "..."
            
        return obj

    def repair_json(self, raw_json: str, schema: Type[T]) -> Any:
        try:
            data = json.loads(raw_json)
        except Exception:
            cleaned = self._pre_process_json_string(raw_json)
            data = json.loads(cleaned)

        if isinstance(data, list) and self._is_pydantic_base(schema):
            base_model = self._get_base_model(schema)
            return [self._repair_align(item, base_model, i) for i, item in enumerate(data)]
        
        final_data = self._repair_align(data, schema)
        
        if "scenes" in final_data and isinstance(final_data["scenes"], list):
            from src.core.models import Scene
            final_data["scenes"] = [self._repair_align(s, Scene, i) for i, s in enumerate(final_data["scenes"])]
            
        return final_data

    def _repair_flatten(self, o: Any) -> str:
        if isinstance(o, str): return o
        if isinstance(o, (int, float, bool)): return str(o)
        if isinstance(o, dict):
            return "; ".join([f"{k}: {self._repair_flatten(v)}" for k, v in o.items()])
        if isinstance(o, list):
            return " | ".join([self._repair_flatten(i) for i in o])
        return str(o)

    def _repair_nested(self, val: Any, annot: Any) -> Any:
        if self._is_pydantic_base(annot):
            return self._repair_align(val, self._get_base_model(annot))
        
        origin = get_origin(annot)
        base_annot = origin or annot

        if base_annot is str and not isinstance(val, str):
            return self._repair_flatten(val)

        if origin is list:
            args = get_args(annot)
            if args and self._is_pydantic_base(args[0]) and isinstance(val, list):
                return [self._repair_align(item, self._get_base_model(args[0])) for item in val]
            
            is_dict = False
            try: is_dict = issubclass(args[0], dict)
            except: pass

            if args and (self._is_pydantic_base(args[0]) or is_dict) and isinstance(val, list):
                target_fields_sub = []
                if is_dict:
                    target_fields_sub = ["speaker", "text"]
                else:
                    target_fields_sub = self._get_base_model(args[0]).model_fields
                
                if "speaker" in target_fields_sub and "text" in target_fields_sub:
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
        return val

    def _repair_align(self, current_data: Any, current_schema: Type[BaseModel], index: Optional[int] = None) -> Any:
        if not isinstance(current_data, dict):
            return current_data

        new_data = {}
        target_fields = current_schema.model_fields
        
        for k, v in current_data.items():
            if k in target_fields:
                target_key = k
            elif k.lower() in self.field_aliases:
                target_key = self.field_aliases[k.lower()]
                if target_key not in target_fields: target_key = k
            else:
                target_key = k
                for target in target_fields:
                    if target.lower() == k.lower():
                        target_key = target
                        break
            
            field_info = target_fields.get(target_key)
            if field_info:
                processed_val = self._repair_nested(v, field_info.annotation)

                if target_key in new_data and isinstance(processed_val, str) and isinstance(new_data[target_key], str):
                    if processed_val not in new_data[target_key]:
                        new_data[target_key] = f"{new_data[target_key]}; {processed_val}"
                else:
                    new_data[target_key] = processed_val

                # Heuristics
                try:
                    base_annot = get_origin(field_info.annotation) or field_info.annotation
                    if base_annot in (int, float):
                        cv = v
                        if isinstance(v, dict):
                            scores = [float(x) for x in v.values() if isinstance(x, (int, float, str)) and str(x).replace('.','',1).isdigit()]
                            if scores: cv = sum(scores) / len(scores)
                        if isinstance(v, str):
                            if "/" in v:
                                try:
                                    n, d = v.split("/")
                                    cv = float(n) / float(d) * 10.0
                                except: pass
                            else:
                                digits = re.findall(r'[-+]?\d*\.\d+|\d+', v)
                                if digits: cv = float(digits[0])
                        if isinstance(cv, (int, float)) and cv > 10 and "score" in target_key.lower():
                            cv = cv / 10.0
                        new_data[target_key] = int(cv) if base_annot is int else float(cv)
                    
                    if base_annot is bool:
                        curr = new_data[target_key]
                        if isinstance(curr, str):
                            vn = curr.lower().strip()
                            if vn in ("yes", "true", "y", "1", "pass", "partially", "mostly"): new_data[target_key] = True
                            elif vn in ("no", "false", "n", "0", "fail", "failed"): new_data[target_key] = False
                except: pass

        # Post-alignment fallbacks
        if "id" in target_fields and new_data.get("id") is None and index is not None:
            new_data["id"] = index + 1

        if "location" in target_fields and not new_data.get("location"):
            for fb in ["setting", "title", "description", "scene"]:
                if current_data.get(fb) and len(str(current_data.get(fb))) < 100:
                    new_data["location"] = str(current_data.get(fb))
                    break
            if not new_data.get("location"): new_data["location"] = "Unknown Location"

        if "narrative_summary" in target_fields and not new_data.get("narrative_summary"):
            for fb in ["scene", "description", "text", "summary"]:
                if current_data.get(fb):
                    new_data["narrative_summary"] = str(current_data.get(fb))
                    break

        if "personality" in target_fields and not new_data.get("personality"):
            new_data["personality"] = "Determined Character"

        if "pronouns" in target_fields and not new_data.get("pronouns"):
            new_data["pronouns"] = "They/Them"

        if "style_guide" in target_fields and isinstance(current_data.get("style_guide"), dict):
            new_data["style_guide"] = "; ".join([f"{ks}: {vs}" for ks, vs in current_data.get("style_guide").items()])

        if "passed" in target_fields and isinstance(current_data.get("passed"), dict):
            new_data["passed"] = all(current_data.get("passed").values())

        return new_data

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
