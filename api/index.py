from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SubmitRequest(BaseModel):
    story_text: str

@app.post("/api/submit")
async def submit_story(req: SubmitRequest):
    # In a pure Vercel deployment we cannot run heavy GPU jobs here.
    # The endpoint simply acknowledges receipt and could enqueue a job.
    return JSONResponse(content={"status": "queued", "job_id": "demo-123"})

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
