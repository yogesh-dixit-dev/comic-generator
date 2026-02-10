import os
import sys
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

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
from src.core.checkpoint import PipelineState
from src.utils.checkpoint_manager import CheckpointManager
from src.agents.infrastructure.telemetry_agent import TelemetryAgent

# Configure logging
log_file = "pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
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
        
        # --- Checkpoint System Initialization ---
        checkpoint_mgr = CheckpointManager(storage)
        input_hash = checkpoint_mgr.get_input_hash(args.input)
        
        # Step 2: Write Script (Chunked for long stories)
        from src.utils.script_consolidator import ScriptConsolidator
        consolidator = ScriptConsolidator()
        
        # Step 1: Parallel Narrative Processing
        logger.info(f"📄 Processing story ({len(raw_text)} characters)...")
        chunks = input_reader.chunk_text(raw_text, max_words=2000)
        logger.info(f"🧩 Split story into {len(chunks)} chunk(s).")
        
        # Limit concurrency for local models to prevent resource saturation
        is_local_reasoning = "ollama" in args.reasoning_model or "local" in args.reasoning_model
        max_narrative_workers = 2 if is_local_reasoning else 5
        
        # Check for checkpoint
        state = checkpoint_mgr.load_checkpoint(input_hash)
        if state:
            # Assuming state.processed_chunks and state.processed_scenes are tracked in PipelineState
            # If not, this logging line might need adjustment based on actual state structure
            logger.info(f"🔄 Resuming from checkpoint. Last chunk index: {state.last_chunk_index}, Master script scenes: {len(state.master_script.scenes) if state.master_script else 0}.")
        else:
            state = PipelineState(input_hash=input_hash)
        
        if state.master_script:
            consolidator.master_script = state.master_script
            consolidator.total_scenes = len(state.master_script.scenes)
            logger.info(f"📜 Restored master script with {consolidator.total_scenes} scenes.")

        # Parallel Script Generation
        chunk_tasks = []
        with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
            for i, chunk_text in enumerate(chunks):
                if i <= state.last_chunk_index:
                    logger.info(f"⏭️ Skipping already processed Chunk {i+1}/{len(chunks)}.")
                    continue
                
                logger.info(f"✍️ Scheduling Chunk {i+1}/{len(chunks)} for parallel processing...")
                task = executor.submit(script_writer.run, chunk_text, expected_schema=ComicScript)
                chunk_tasks.append((i, task))

        # Gather results and maintain order
        for i, task in chunk_tasks:
            try:
                chunk_script = task.result()
                consolidator.add_chunk(chunk_script)
                
                # Update State & Save Checkpoint
                state.last_chunk_index = i
                state.master_script = consolidator.get_script()
                checkpoint_mgr.save_checkpoint(state)
            except Exception as e:
                logger.error(f"❌ Failed to process parallel chunk {i+1}: {e}")
                if i == 0: raise 

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
        if state.characters:
            characters = state.characters
            logger.info(f"⏭️ Skipping character design (Restored {len(characters)} characters).")
        else:
            characters = character_designer.run(script) 
            logger.info(f"👤 Designed {len(characters)} characters.")
            
            char_critique = character_critique.run(characters)
            if not char_critique.passed:
                logger.warning(f"Character Critique Failed: {char_critique.feedback}")
            
            # Save Checkpoint
            state.characters = characters
            checkpoint_mgr.save_checkpoint(state)

        # Step 5: Visual Production (Scene by Scene)
        finished_pages = []
        for scene in script.scenes:
            if scene.id <= state.last_scene_id:
                logger.info(f"⏭️ Skipping already produced Scene {scene.id}.")
                # (Optional: Load previously rendered images if needed for layout engine later)
                # For now, we assume the layout engine re-assembles or we need to persist finished_pages too
                continue

            logger.info(f"🎬 Producing Scene {scene.id}: {scene.location}")
            
            # Director plans the shots
            scene_plan = director.run(scene)
            
            # Illustrator generates images (Batch Optimized)
            illustrator.run_batch(scene_plan.panels, characters=characters, style_guide=script.style_guide)
            
            # Step 6: Layout & Lettering (Per Scene for now, or Per Page)
            logger.info(f"📐 Assembling Scene {scene.id}...")
            # Note: layout_engine takes a list of panels and returns panels with layout metadata
            scene_panels = layout_engine.run(scene_plan.panels)
            
            # Lettering adds text to the images (Parallelized for speed)
            lettering_tasks = []
            with ThreadPoolExecutor(max_workers=min(4, len(scene_panels))) as executor:
                for panel in scene_panels:
                    task = executor.submit(lettering_agent.run, panel)
                    lettering_tasks.append(task)
            
            for task in lettering_tasks:
                final_image_path = task.result()
                if final_image_path:
                    state.finished_pages.append(final_image_path)
            
            # Save Checkpoint after each scene (Thread pool manages this in background)
            state.last_scene_id = scene.id
            checkpoint_mgr.save_checkpoint(state)
        
        # Step 7: Final Comic Packaging
        logger.info("📦 Packaging final comic...")
        output_paths = storage.save_comic(script, state.finished_pages, args.output)
        logger.info(f"🚀 Comic Generation Complete! Output saved to {args.output}")
        for p in output_paths:
            logger.info(f"  - {p}")
            
        # Optional: Sync to HF if configured
        if args.storage == "hf":
            storage.sync(source_dir=args.output, target_dir="comic_output")
            logger.info("Synced output to Hugging Face Hub successfully.")

        # Step 8: Telemetry & Log Analysis
        logger.info("📊 Running Telemetry Analysis...")
        telemetry_agent = TelemetryAgent("TelemetryObserver")
        report = telemetry_agent.run("pipeline.log")
        
        logger.info("📈 --- Actionable Insights ---")
        for suggestion in report.get("actionable_improvements", []):
            logger.info(f"💡 {suggestion}")
        
        # Ensure all background tasks complete
        logger.info("⏳ Finalizing background tasks...")
        checkpoint_mgr.shutdown()

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists("output"):
        os.makedirs("output")
    main()
