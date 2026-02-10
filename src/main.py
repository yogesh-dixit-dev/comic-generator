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
    
    parser.add_argument("--reasoning_model", type=str, default="ollama/llama3.1", help="Model for complex reasoning tasks")
    parser.add_argument("--fast_model", type=str, default="ollama/llama3.2", help="Model for fast, simple tasks")
    
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

    # Early exit for validation/CI runs
    if os.environ.get("PIPELINE_VALIDATION_RUN"):
        logger.info("Validation run detected. Environment initialized successfully. Exiting.")
        sys.exit(0)

    # 3. Initialize Agents
    input_reader = InputReaderAgent("InputReader")
    
    # Tiered Model Mapping
    reasoning_model = args.reasoning_model
    fast_model = args.fast_model
    
    logger.info(f"🧠 Reasoning Model: {reasoning_model}")
    logger.info(f"⚡ Fast Model: {fast_model}")

    script_writer = ScriptWriterAgent("ScriptWriter", config={"model_name": reasoning_model})
    script_critique = ScriptCritiqueAgent("ScriptCritique", config={"model_name": reasoning_model})
    
    character_designer = CharacterDesignAgent("CharacterDesigner", config={"model_name": reasoning_model})
    character_critique = CharacterCritiqueAgent("CharacterCritique", config={"model_name": fast_model})
    
    consistency_manager = ConsistencyManager("ConsistencyManager")
    director = DirectorAgent("Director", config={"model_name": reasoning_model})
    
    illustrator = IllustratorAgent("Illustrator", image_generator=image_gen, consistency_manager=consistency_manager)
    layout_engine = LayoutEngine("LayoutEngine")
    lettering_agent = LetteringAgent("LetteringAgent", config={"model_name": fast_model})

    # 4. Execution Pipeline
    try:
        # Step 1: Read Input
        raw_text = input_reader.process(args.input)
        
        if not raw_text or not raw_text.strip():
            logger.error(f"❌ Input from '{args.input}' is empty! Please ensure the file has content.")
            return
        
        logger.info(f"📄 Processing story ({len(raw_text)} characters)...")
        
        # Step 2: Write Script (Chunked for long stories)
        from src.utils.script_consolidator import ScriptConsolidator
        consolidator = ScriptConsolidator()
        
        # Decide chunk size. 2000 words is safe for Llama 3.2
        chunks = input_reader.chunk_text(raw_text, max_words=2000)
        logger.info(f"🧩 Split story into {len(chunks)} chunk(s).")

        for i, chunk_text in enumerate(chunks):
            logger.info(f"✍️ Processing Chunk {i+1}/{len(chunks)}...")
            try:
                # We skip schema validation on run if it's too restrictive for multi-chunk
                # But here ComicScript matches.
                chunk_script = script_writer.run(chunk_text, expected_schema=ComicScript)
                consolidator.add_chunk(chunk_script)
            except Exception as e:
                logger.error(f"❌ Failed to process chunk {i+1}: {e}")
                if i == 0: raise 
                continue

        script = consolidator.get_script()
        if not script:
            logger.error("❌ No script scenes were generated!")
            return

        logger.info(f"✅ Generated Full Script: '{script.title}' with {len(script.scenes)} scenes.")
        
        # Step 3: Critique Script
        critique = script_critique.run(script)
        if not critique.passed:
            logger.warning(f"Script Critique Failed: {critique.feedback}")
        
        # Step 4: Character Design
        characters = character_designer.run(script) 
        logger.info(f"👤 Designed {len(characters)} characters.")
        
        char_critique = character_critique.run(characters)
        if not char_critique.passed:
             logger.warning(f"Character Critique Failed: {char_critique.feedback}")

        # Step 5: Visual Production (Scene by Scene)
        finished_pages = []
        for scene in script.scenes:
            logger.info(f"🎬 Producing Scene {scene.id}: {scene.location}")
            
            # Director plans the shots
            scene_plan = director.run(scene)
            
            # Illustrator generates images
            for panel in scene_plan.panels:
                logger.info(f"🎨 Illustrating Panel {panel.id}...")
                illustrator.run(panel)
            
            # Step 6: Layout & Lettering (Per Scene for now, or Per Page)
            logger.info(f"📐 Assembling Scene {scene.id}...")
            scene_pages = layout_engine.run(scene.panels)
            
            for page in scene_pages:
                final_page = lettering_agent.run(page)
                finished_pages.append(final_page)
        
        # Step 7: Storage
        output_paths = storage.save_comic(script, finished_pages, args.output)
        logger.info(f"🚀 Comic Generation Complete! Output saved to {args.output}")
        for p in output_paths:
            logger.info(f"  - {p}")
            
        # Optional: Sync to HF if configured
        if args.storage == "hf":
            storage.sync(source_dir=args.output, target_dir="comic_output")
            logger.info("Synced output to Hugging Face Hub successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists("output"):
        os.makedirs("output")
    main()
