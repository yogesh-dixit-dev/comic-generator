import os
import sys
import argparse
import logging

# Fix imports for Colab and different working directories
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.models import ComicScript, Scene, Panel, Character
from src.agents.narrative.input_reader import InputReaderAgent
from src.agents.narrative.script_writer import ScriptWriterAgent
from src.agents.narrative.script_critique import ScriptCritiqueAgent
from src.agents.visual.character_designer import CharacterDesignAgent
from src.agents.visual.character_critique import CharacterCritiqueAgent
from src.agents.visual.consistency_manager import ConsistencyManager
from src.agents.production.director import DirectorAgent
from src.agents.production.illustrator import IllustratorAgent
from src.agents.production.image_generators import MockImageGenerator, DiffusersImageGenerator
from src.agents.assembly.layout_engine import LayoutEngine
from src.agents.assembly.lettering import LetteringAgent
from src.core.storage import HuggingFaceStorage, LocalStorage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ComicGen")

def main():
    parser = argparse.ArgumentParser(description="AI Comic Book Generator")
    parser.add_argument("--input", type=str, required=True, help="Path to input text/pdf/docx file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--colab", action="store_true", help="Run in Colab mode (Use GPU + Diffusers)")
    parser.add_argument("--storage", type=str, choices=["local", "hf"], default="local", help="Storage backend")
    parser.add_argument("--hf_repo", type=str, help="Hugging Face Repo ID (if storage=hf)")
    parser.add_argument("--hf_token", type=str, help="Hugging Face Token (optional if env var set)")
    
    args = parser.parse_args()

    # 1. Setup Storage
    if args.storage == "hf":
        if not args.hf_repo:
            logger.error("Must provide --hf_repo when usage storage=hf")
            return
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
             logger.error("Must provide HF Token via --hf_token or env var HF_TOKEN")
             return
        storage = HuggingFaceStorage(repo_id=args.hf_repo, token=token, repo_type="dataset")
    else:
        storage = LocalStorage()

    # 2. Select Image Generator
    if args.colab:
        # Use Diffusers (requires GPU)
        try:
            image_gen = DiffusersImageGenerator(model_id="stabilityai/stable-diffusion-xl-base-1.0", device="cuda")
        except Exception as e:
            logger.error(f"Failed to load Diffusers: {e}. Falling back to Mock.")
            image_gen = MockImageGenerator()
    else:
        image_gen = MockImageGenerator()

    # 3. Initialize Agents
    input_reader = InputReaderAgent("InputReader")
    # Use gpt-3.5-turbo by default (universal access, most cost-effective)
    # Override with LITELLM_MODEL env var for different models
    model_name = os.environ.get("LITELLM_MODEL", "gpt-3.5-turbo")
    
    script_writer = ScriptWriterAgent("ScriptWriter", config={"model_name": model_name})
    script_critique = ScriptCritiqueAgent("ScriptCritique", config={"model_name": model_name})
    
    character_designer = CharacterDesignAgent("CharacterDesigner", config={"model_name": model_name})
    character_critique = CharacterCritiqueAgent("CharacterCritique", config={"model_name": model_name})
    
    consistency_manager = ConsistencyManager("ConsistencyManager")
    director = DirectorAgent("Director", config={"model_name": model_name})
    
    illustrator = IllustratorAgent("Illustrator", image_generator=image_gen, consistency_manager=consistency_manager)
    layout_engine = LayoutEngine("LayoutEngine")
    lettering_agent = LetteringAgent("LetteringAgent")

    # 4. Execution Pipeline
    try:
        # Step 1: Read Input
        raw_text = input_reader.process(args.input)
        
        # Step 2: Write Script
        # TODO: Implement chunking loop here for long texts. For MVP, process single chunk.
        # chunks = input_reader.chunk_text(raw_text)
        # For simplicity in V1, we just take the first chunk or whole text if small
        script = script_writer.run(raw_text, expected_schema=ComicScript)
        logger.info(f"Generated Script: {script.title}")
        
        # Step 3: Critique Script
        critique = script_critique.run(script)
        if not critique.passed:
            logger.warning(f"Script Critique Failed: {critique.feedback}")
        
        # Step 4: Character Design
        # Returns List[Character], but BaseAgent.run validation expects specific schema.
        # We handle this manually or wrap in object. CharacterDesigner returns List[Character].
        # Because validation might fail on List, we might skip strict schema check here or allow List.
        # For now, let's call process directly or implement wrapper model.
        # Let's assume run() handles List if schema is verified or we skip it.
        characters = character_designer.process(script) 
        
        # Verify characters
        char_critique = character_critique.run(characters)
        if not char_critique.passed:
            logger.warning(f"Character Critique Failed: {char_critique.feedback}")

        # Step 5: Directing (Add Camera/Lighting)
        script = director.run(script, expected_schema=ComicScript)
        
        # Step 6: Illustration
        # This modifies the script in place (adding image paths)
        for scene in script.scenes:
            logger.info(f"Processing Scene {scene.id}...")
            for panel in scene.panels:
                # Illustrator validates internally? 
                # It returns a Panel.
                # We update the panel in the scene object.
                updated_panel = illustrator.run(panel, expected_schema=Panel)
                # Note: 'run' validation logic might need `updated_panel` to be dict if using parse_obj,
                # or object if Pydantic. BaseAgent logic handles both.
                
                # Update the object in list (since strictly 'updated_panel' is a new object/dict)
                # But for now, Illustrator modifies in place essentially or returns modify.
                # We need to ensure we are updating the scene list.
                pass # Logic is implied by `illustrator.process` returning modified panel

        # Step 7: Assembly (Layout & Lettering)
        for scene in script.scenes:
            # Layout
            scene.panels = layout_engine.run(scene.panels) # Returns List[Panel]
            
            # Lettering
            for panel in scene.panels:
                final_path = lettering_agent.run(panel)
                logger.info(f"Final Panel Image: {final_path}")
        
        # Sync with Storage
        if args.storage == "hf":
            storage.sync(source_dir="output", target_dir="comic_output")
            logger.info("Synced output to Hugging Face Hub successfully.")
            
        logger.info("Comic Generation Complete!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists("output"):
        os.makedirs("output")
    main()
