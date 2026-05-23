import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/submit")
async def submit_story(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    dest_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(dest_path, "wb") as out:
        content = await file.read()
        out.write(content)
    # Placeholder for queue integration
    return JSONResponse(content={"status": "queued", "job_id": job_id, "path": dest_path})
