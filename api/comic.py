from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os

app = FastAPI()

# Simple CORS configuration (Vercel will add the header automatically for same origin)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

class SubmitRequest(BaseModel):
    story_text: str
    # additional fields like options can be added later

@app.post("/api/submit")
async def submit_story(req: SubmitRequest):
    # In a pure Vercel setup we cannot run heavy GPU work here.
    # Instead we just acknowledge and could push the request to a queue.
    # For now we simulate immediate acceptance.
    return JSONResponse(content={"status": "queued", "job_id": "demo-123"})

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
