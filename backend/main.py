import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

# Allow the Vercel frontend origin (or any origin during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "*")],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SubmitRequest(BaseModel):
    story_text: str

@app.post("/api/submit")
async def submit_story(req: SubmitRequest):
    # In the decoupled architecture we just acknowledge receipt.
    # A real implementation would enqueue a job for heavy GPU work.
    return JSONResponse(content={"status": "queued", "job_id": "demo-123"})

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000))
