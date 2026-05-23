from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys
import json
import hashlib
import logging
import subprocess
from typing import Dict, Any, List, Optional
from src.core.checkpoint import PipelineState
from src.utils.checkpoint_manager import CheckpointManager
from src.core.storage import LocalStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ComicServer")

app = FastAPI(title="Comic Gen HITL Server")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage initialization
storage = LocalStorage()
checkpoint_mgr = CheckpointManager(storage)

# Keep track of active pipeline processes
active_processes: Dict[str, subprocess.Popen] = {}

# Ensure upload directory exists
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)

@app.get("/api/state/{input_hash}")
async def get_state(input_hash: str):
    """Retrieves the current pipeline state for a given input hash."""
    state = checkpoint_mgr.load_checkpoint(input_hash)
    if not state:
        return {"error": "Checkpoint not found"}
    
    # Enrich state with active process running status
    state_dict = state.dict()
    is_running = False
    proc = active_processes.get(input_hash)
    if proc:
        if proc.poll() is None:
            is_running = True
        else:
            # Clean up finished process
            active_processes.pop(input_hash, None)
            
    state_dict["metadata"]["is_running"] = is_running
    return state_dict

@app.post("/api/state/{input_hash}/update")
async def update_state(input_hash: str, updated_data: Dict[str, Any]):
    """Updates the pipeline state (used for HITL edits)."""
    state = checkpoint_mgr.load_checkpoint(input_hash)
    if not state:
         raise HTTPException(status_code=404, detail="Checkpoint not found")
    
    state_dict = state.dict()
    
    # Handle deep merge of metadata, master_script and characters
    if "metadata" in updated_data and isinstance(updated_data["metadata"], dict):
        if "metadata" not in state_dict or state_dict["metadata"] is None:
            state_dict["metadata"] = {}
        state_dict["metadata"].update(updated_data["metadata"])
        
    if "master_script" in updated_data and isinstance(updated_data["master_script"], dict):
        if "master_script" not in state_dict or state_dict["master_script"] is None:
             state_dict["master_script"] = {}
        state_dict["master_script"].update(updated_data["master_script"])
        
    if "characters" in updated_data and isinstance(updated_data["characters"], list):
        state_dict["characters"] = updated_data["characters"]
        
    if "scene_plans" in updated_data and isinstance(updated_data["scene_plans"], dict):
        if "scene_plans" not in state_dict or state_dict["scene_plans"] is None:
             state_dict["scene_plans"] = {}
        state_dict["scene_plans"].update(updated_data["scene_plans"])

    # Update other top-level fields if passed
    for k, v in updated_data.items():
        if k not in ["metadata", "master_script", "characters", "scene_plans"]:
            state_dict[k] = v

    new_state = PipelineState.model_validate(state_dict)
    checkpoint_mgr.save_checkpoint(new_state)
    return {"status": "success", "state": new_state.dict()}

@app.post("/api/pipeline/start")
async def start_pipeline(req: Request):
    """
    Accepts text input, creates the upload file and checkpoint,
    and launches the Python pipeline process in the background.
    """
    data = await req.json()
    story_text = data.get("story_text", "").strip()
    project_name = data.get("project_name", "").strip() or "Unnamed Comic"
    style_preset = data.get("style_preset", "").strip()
    auto_run = data.get("auto_run", True)
    
    if not story_text:
        raise HTTPException(status_code=400, detail="Story text is required")
        
    # Calculate Blake2b hash of story text
    hasher = hashlib.blake2b()
    hasher.update(story_text.encode("utf-8"))
    story_hash = hasher.hexdigest()
    
    # Save the text file
    file_path = os.path.join("uploads", f"story_{story_hash}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(story_text)
        
    # Pre-create or load checkpoint
    state = checkpoint_mgr.load_checkpoint(story_hash)
    if not state:
        state = PipelineState(
            input_hash=story_hash,
            metadata={
                "project_name": project_name,
                "auto_run": auto_run,
                "style_preset": style_preset,
                "current_step": "initializing"
            }
        )
    else:
        # Update existing metadata
        state.metadata["project_name"] = project_name
        state.metadata["auto_run"] = auto_run
        if style_preset:
            state.metadata["style_preset"] = style_preset
            
    checkpoint_mgr.save_checkpoint(state)
    
    # Check if a process is already running for this hash
    existing_proc = active_processes.get(story_hash)
    if existing_proc and existing_proc.poll() is None:
         logger.info(f"Pipeline already running for hash {story_hash[:12]}")
         return {"status": "running", "input_hash": story_hash}
         
    # Launch pipeline process in the background
    cmd = [
        sys.executable,
        "src/main.py",
        "--input", file_path,
        "--colab"
    ]
    
    # Append styling to overall style_guide in main.py by writing to script if style_preset given
    logger.info(f"Launching subprocess: {' '.join(cmd)}")
    
    # Spawn in background, redirect outputs to file
    log_file = open("pipeline.log", "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    active_processes[story_hash] = proc
    return {"status": "started", "input_hash": story_hash}

@app.post("/api/pipeline/stop/{input_hash}")
async def stop_pipeline(input_hash: str):
    """Terminates an active pipeline process and sets auto_run = False."""
    proc = active_processes.get(input_hash)
    if proc:
        logger.info(f"Stopping pipeline process for hash {input_hash[:12]}...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        active_processes.pop(input_hash, None)
        
    state = checkpoint_mgr.load_checkpoint(input_hash)
    if state:
        state.metadata["auto_run"] = False
        checkpoint_mgr.save_checkpoint(state)
        
    return {"status": "stopped"}

@app.post("/api/state/{input_hash}/approve/{step}")
async def approve_step(input_hash: str, step: str, updated_data: Optional[Dict[str, Any]] = None):
    """
    Saves optional updated edits first, then sets approved_step = True
    in metadata to resume the blocked python agent.
    """
    state = checkpoint_mgr.load_checkpoint(input_hash)
    if not state:
         raise HTTPException(status_code=404, detail="Checkpoint not found")
         
    # Merge any incoming UI edits first
    if updated_data:
        state_dict = state.dict()
        if "master_script" in updated_data and isinstance(updated_data["master_script"], dict):
            if "master_script" not in state_dict or state_dict["master_script"] is None:
                state_dict["master_script"] = {}
            state_dict["master_script"].update(updated_data["master_script"])
        if "characters" in updated_data and isinstance(updated_data["characters"], list):
            state_dict["characters"] = updated_data["characters"]
        if "scene_plans" in updated_data and isinstance(updated_data["scene_plans"], dict):
            if "scene_plans" not in state_dict or state_dict["scene_plans"] is None:
                state_dict["scene_plans"] = {}
            state_dict["scene_plans"].update(updated_data["scene_plans"])
        state = PipelineState.model_validate(state_dict)

    # Set approved flag
    state.metadata[f"approved_{step}"] = True
    checkpoint_mgr.save_checkpoint(state)
    logger.info(f"Set approved_{step} = True for {input_hash[:12]}")
    return {"status": "success", "state": state.dict()}

@app.post("/api/state/{input_hash}/toggle-auto-run")
async def toggle_auto_run(input_hash: str, req: Request):
    """Toggles uninterrupted auto run status in checkpoint."""
    data = await req.json()
    auto_run = data.get("auto_run", True)
    
    state = checkpoint_mgr.load_checkpoint(input_hash)
    if not state:
         raise HTTPException(status_code=404, detail="Checkpoint not found")
         
    state.metadata["auto_run"] = auto_run
    
    # If the user toggles auto_run ON, and the pipeline is currently paused,
    # let's release the current pause step automatically
    if auto_run:
        current_step = state.metadata.get("current_step")
        if current_step:
            state.metadata[f"approved_{current_step}"] = True
            
    checkpoint_mgr.save_checkpoint(state)
    return {"status": "success", "auto_run": auto_run}

@app.get("/api/projects")
async def list_projects():
    """Lists all available project checkpoints with dynamic process state."""
    projects_list = checkpoint_mgr.list_checkpoints()
    for project in projects_list:
        h = project["input_hash"]
        is_running = False
        proc = active_processes.get(h)
        if proc:
            if proc.poll() is None:
                is_running = True
            else:
                active_processes.pop(h, None)
        project["is_running"] = is_running
    return projects_list

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

# Serve generated static panel assets
app.mount("/output", StaticFiles(directory="output"), name="output")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
