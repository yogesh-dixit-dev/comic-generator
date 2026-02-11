from fastapi import FastAPI, WebSocket, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import json
import logging
from typing import Dict, Any, List
from src.core.checkpoint import PipelineState
from src.utils.checkpoint_manager import CheckpointManager
from src.core.storage import LocalStorage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComicServer")

app = FastAPI(title="Comic Gen HITL Server")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage initialization
storage = LocalStorage()
checkpoint_mgr = CheckpointManager(storage)

@app.get("/api/state/{input_hash}")
async def get_state(input_hash: str):
    """Retrieves the current pipeline state for a given input hash."""
    state = checkpoint_mgr.load_checkpoint(input_hash)
    if not state:
        return {"error": "Checkpoint not found"}
    return state.dict()

@app.post("/api/state/{input_hash}/update")
async def update_state(input_hash: str, updated_data: Dict[str, Any]):
    """Updates the pipeline state (used for HITL edits)."""
    state = checkpoint_mgr.load_checkpoint(input_hash)
    if not state:
        return {"error": "Checkpoint not found"}
    
    # Merge updated data into state
    # This is a simple merge; complex fields might need more care
    state_dict = state.dict()
    state_dict.update(updated_data)
    new_state = PipelineState.model_validate(state_dict)
    
    checkpoint_mgr.save_checkpoint(new_state)
    return {"status": "success", "state": new_state.dict()}

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

# Static files for generated images
# Assuming output directory is 'output'
if os.path.exists("output"):
    app.mount("/output", StaticFiles(directory="output"), name="output")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
