import os
import sys

# Set PyTorch memory allocation strategy to prevent fragmentation.
# This MUST be set before torch is imported or any CUDA calls are made.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    parser.add_argument("--phase", type=str, choices=["all", "plan", "draw"], default="all", help="Execution phase: all (default), plan (narrative+visual planning), or draw (isolated image generation)")
    
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

    # 2. Select Image Generator (Initially Mock to save VRAM during LLM planning)
    # Real generator (Diffusers) will be loaded in the visual production phase if --colab is set.
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
        # Step 0: Pre-flight Health Checks
        if args.phase in ["all", "plan"]:
            logger.info("🛡️ Running pre-flight health checks...")
            from src.utils.llm_interface import LLMInterface
            reasoning_llm = LLMInterface(model_name=args.reasoning_model)
            
            if not reasoning_llm.is_healthy():
                logger.error("❌ LOCAL LLM SERVICE (Ollama) IS UNREACHABLE!")
                logger.info("💡 Potential Fixes:")
                logger.info("   1. Colab: Run '!ollama serve &' in a new cell.")
                logger.info("   2. Local: Ensure Ollama is running on port 11434.")
                logger.info("   3. Check: Open http://localhost:11434/ in a browser.")
                return

            logger.info("✅ Core LLM backend is healthy.")
        else:
            logger.info("⏭️ Skipping LLM health checks (Independent Drawing Mode).")
            from src.utils.llm_interface import LLMInterface
            reasoning_llm = LLMInterface(model_name=args.reasoning_model)

        # Step 1: Input Analysis
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
                
                # Progress Update: Planning - Scripting (0-40%)
                progress = int(((i + 1) / len(chunks)) * 40)
                logger.info(f"[PROGRESS] {progress}% - Scripting chunk {i+1}/{len(chunks)} complete.")
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
        if critique and not critique.passed:
            logger.warning(f"Script Critique Failed: {critique.feedback}")
        elif not critique:
            logger.error("❌ Script critique failed to return a result!")
        
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
            logger.info("[PROGRESS] 50% - Character design complete.")

        # Step 5: Visual Planning Phase (LLM Heavy)
        # We pre-calculate all scene plans BEFORE starting image generation to avoid VRAM competition.
        if args.phase in ["all", "plan"]:
            logger.info("📐 Starting Visual Planning Phase...")
            for scene in script.scenes:
                # Check if we should skip planning if scene images already exist
                scene_done = scene.id <= state.last_scene_id
                files_exist = True
                if scene_done:
                    for panel in scene.panels:
                        import hashlib
                        h = hashlib.md5(panel.image_prompt.encode() if panel.image_prompt else "".encode()).hexdigest()[:8]
                        if not os.path.exists(os.path.join("output", f"gen_{h}_lettered.png")):
                            files_exist = False
                            break
                
                if scene_done and files_exist:
                    continue

                if str(scene.id) in state.scene_plans or scene.id in state.scene_plans:
                     logger.info(f"⏭️ Skipping Planning for Scene {scene.id} (already planned).")
                     continue

                logger.info(f"📋 Planning Scene {scene.id}: {scene.location}...")
                state.scene_plans[scene.id] = director.run(scene).dict() # Store as dict for serialization
                
                # Progress Update: Planning - Visual (50-100%)
                scenes_processed = len(state.scene_plans)
                progress = 50 + int((scenes_processed / len(script.scenes)) * 50)
                logger.info(f"[PROGRESS] {progress}% - visual Planning scene {scenes_processed}/{len(script.scenes)} complete.")
                
            # Save Checkpoint after planning all scenes
            checkpoint_mgr.save_checkpoint(state)
            
            if args.phase == "plan":
                logger.info("🏁 Planning Phase Complete. Exiting to free GPU for drawing.")
                return

        # Step 6: GPU Isolate Phase (Cleanup)
        # Force unload all Ollama models from VRAM before starting SDXL.
        if args.phase in ["all", "draw"]:
            logger.info("🧹 Clearing GPU for Image Generation...")
            
            # 1. Unload Thinking/Reasoning models
            reasoning_llm.unload_model()
            
            # 2. Unload Fast/Utility models (for Character Critique/Lettering setup)
            from src.utils.llm_interface import LLMInterface
            fast_llm = LLMInterface(model_name=args.fast_model)
            fast_llm.unload_model()
            
            # 3. Deep Garbage Collection
            import gc
            import time
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("✅ GPU Isolated. Waiting 2s for VRAM to settle...")
            time.sleep(2)

        # Step 7: Visual Production Phase (GPU Heavy)
        if args.phase in ["all", "draw"]:
            # Delay loading SDXL until LLMs are fully unloaded from VRAM.
            if args.colab and not isinstance(image_gen, DiffusersImageGenerator):
                logger.info("🚀 Loading Diffusers Image Generator (Isolated Mode)...")
                try:
                    # Reload the generator in isolated context
                    image_gen = DiffusersImageGenerator(model_id="stabilityai/stable-diffusion-xl-base-1.0", device="cuda")
                    illustrator.image_generator = image_gen
                except Exception as e:
                    logger.error(f"Failed to load Diffusers: {e}. Falling back to Mock.")
                    image_gen = MockImageGenerator()
                    illustrator.image_generator = image_gen

            finished_pages = []
            for scene in script.scenes:
                # Retrieve plan from state
                scene_plan_data = state.scene_plans.get(scene.id) or state.scene_plans.get(str(scene.id))
                
                if not scene_plan_data:
                    # Robust Resume: If plan is missing but we're in draw mode, we might need to re-plan or error
                    if args.phase == "draw":
                        logger.warning(f"⚠️ Plan missing for Scene {scene.id}. Re-planning in Draw mode...")
                        scene_plan_data = director.run(scene).dict()
                    else:
                         continue

                logger.info(f"🎨 Drawing Scene {scene.id}...")
                # Re-validate model if it was stored as dict
                from src.core.models import ScenePlan
                scene_plan = ScenePlan.model_validate(scene_plan_data)
                
                # Illustrator generates images (Batch Optimized)
                illustrator.run_batch(scene_plan.panels, characters=characters, style_guide=script.style_guide)
            
            # Step 8: Layout & Lettering (Per Scene for now, or Per Page)
            logger.info(f"📐 Assembling Scene {scene.id}...")
            scene_panels = layout_engine.run(scene_plan.panels)
            
            # Lettering adds text to the images
            lettering_tasks = []
            with ThreadPoolExecutor(max_workers=min(4, len(scene_panels))) as executor:
                for panel in scene_panels:
                    task = executor.submit(lettering_agent.run, panel)
                    lettering_tasks.append(task)
            
            for task in lettering_tasks:
                final_image_path = task.result()
                if final_image_path:
                    state.finished_pages.append(final_image_path)
            
            # Save Checkpoint after each scene
            state.last_scene_id = scene.id
            checkpoint_mgr.save_checkpoint(state)
            
            # Progress Update: Drawing (0-100% of Draw phase, but logged for visibility)
            scenes_drawn = len([s for s in script.scenes if s.id <= state.last_scene_id])
            progress = int((scenes_drawn / len(script.scenes)) * 100)
            logger.info(f"[PROGRESS] {progress}% - Drawing scene {scenes_drawn}/{len(script.scenes)} complete.")
        
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
