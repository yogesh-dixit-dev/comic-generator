from typing import List, Dict, Any
from src.core.agent import BaseAgent
from src.core.models import Character, ComicScript
from src.utils.llm_interface import LLMInterface

class CharacterDesignAgent(BaseAgent):
    def __init__(self, agent_name: str = "CharacterDesigner", config: Dict[str, Any] = None):
        super().__init__(agent_name, config)
        self.llm = LLMInterface(model_name=config.get("model_name", "gpt-4o"))

    def process(self, script: ComicScript) -> List[Character]:
        """
        Analyzes the script and generates detailed, unique character profiles.
        """
        self.logger.info("Generating character design profiles...")
        
        system_prompt = """
        You are a master character designer for comics and animation.
        Your task is to create detailed visual profiles for each main character in the script.
        
        CRITICAL RULES:
        1. UNIQUENESS: Do not create duplicate profiles for the same person.
        2. ALIASES: Identify if the same character is referred to by different names (e.g. 'Santiago' and 'The Shepherd'). List all such names in the 'aliases' field.
        3. PRONOUNS: Specify the correct pronouns for the character.
        4. CONSISTENCY: Ensure the visual descriptions are detailed enough for an illustrator to maintain consistency across different scenes.
        
        For each character, specify:
        - name: The unique, canonical name.
        - aliases: List of other ways they are mentioned.
        - pronouns: e.g. 'He/Him', 'She/Her', 'They/Them'.
        - description: Physical traits (hair, eyes, build, clothing style, distinct features like scars/glasses).
        - personality: Traits that affect posture and expression.
        
        The output must be a list of Character objects.
        """
        
        # Extract character names from the script scenes to guide the LLM
        character_names = set()
        for scene in script.scenes:
            if scene.characters:
                for char in scene.characters:
                    character_names.add(char.name)
            if scene.panels:
                 for panel in scene.panels:
                     if panel.characters_present:
                         for name in panel.characters_present:
                             character_names.add(name)

        user_prompt = f"""
        Analyze the following script and create character profiles for these characters: {', '.join(character_names)}
        
        Script Title: {script.title}
        Script Synopsis: {script.synopsis}
        
        (Infer details from the context of the story if not explicitly stated).
        Make sure to merge similar characters into one with aliases.
        """
        
        from pydantic import BaseModel
        class CharacterList(BaseModel):
            characters: List[Character]
        
        try:
            result = self.llm.generate_structured_output(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=CharacterList
            )
            
            # Smart Deduplication & Alias Merging
            characters_by_name = {}
            
            for char in result.characters:
                # 1. Normalize name
                canonical = char.name.strip().lower()
                
                # 2. Check if this character is already known via an alias
                existing_char = None
                for known_char in characters_by_name.values():
                    # If current char name matches an existing char's alias OR vice versa
                    if canonical == known_char.name.lower() or \
                       any(a.lower() == canonical for a in known_char.aliases) or \
                       any(a.lower() == known_char.name.lower() for a in char.aliases):
                        existing_char = known_char
                        break
                
                if existing_char:
                    # Merge logic: Combine aliases and append descriptions
                    existing_char.aliases = list(set(existing_char.aliases + char.aliases + [char.name]))
                    if char.description not in existing_char.description:
                        existing_char.description += f" | {char.description}"
                    if char.personality not in existing_char.personality:
                        existing_char.personality += f" ; {char.personality}"
                    self.logger.info(f"Merged duplicate character profile for '{char.name}' into '{existing_char.name}'.")
                else:
                    characters_by_name[canonical] = char
            
            return list(characters_by_name.values())
        except Exception as e:
            self.logger.error(f"Failed to generate character profiles: {e}")
            raise
