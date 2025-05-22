import os
import uuid
import shutil
from fastapi import File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import time
from app import app

from producers.jdilla import apply_j_dilla_effect
from producers.steve_albini import apply_steve_albini_effect
from producers.scott_burns import apply_scott_burns_effect

# Use Railway volume paths for storage

RAILWAY_VOLUME = os.getenv('RAILWAY_VOLUME_MOUNT_PATH', 'server/data')
UPLOAD_DIR = os.path.join(RAILWAY_VOLUME, "uploads")
PROCESSED_DIR = os.path.join(RAILWAY_VOLUME, "processed")


# Constants for audio processing
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for file handling
MAX_AUDIO_LENGTH = 600  # Maximum audio length in seconds
SAMPLE_RATE = 44100  # Standard sample rate


# Create directories in Railway volume
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "J Dilla Remix API"}


@app.post("/process-audio")
async def process_audio(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...)
):
    """
    Upload an MP3 file and process it to sound like J Dilla
    """
    if not audio.content_type or "audio/mpeg" not in audio.content_type:
        raise HTTPException(
            status_code=400, 
            detail="Only MP3 files are accepted"
        )
    
    task_id = str(uuid.uuid4())
    file_extension = os.path.splitext(audio.filename)[1] if audio.filename else ".mp3"
    
    input_path = os.path.join(UPLOAD_DIR, f"{task_id}{file_extension}")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    for style in ['dilla', 'albini', 'burns']:
        output_path = os.path.join(PROCESSED_DIR, f"{task_id}_{style}{file_extension}")
        if style == 'dilla':
            proc_func = apply_j_dilla_effect
        elif style == 'albini':
            proc_func = apply_steve_albini_effect
        else:
            proc_func = apply_scott_burns_effect
    
        background_tasks.add_task(
            proc_func,
            input_path=input_path,
            output_path=output_path,
            task_id=task_id
        )
    
    time.sleep(2)  # Simulate processing time
    
    return {
        "status": "success",
        "message": "Audio processed successfully",
        "taskId": task_id
    }

@app.get("/audio/{file_id}")
async def get_audio(file_id: str):
    """
    Retrieve a processed audio file by ID
    """
    # Look for the file in the processed directory
    for file in os.listdir(PROCESSED_DIR):
        if file.startswith(file_id):
            file_path = os.path.join(PROCESSED_DIR, file)
            return FileResponse(
                file_path,
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f"attachment; filename={file}"
                }
            )
    
    raise HTTPException(status_code=404, detail="Audio file not found")

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Check the status of a processing task and return the file URL if complete
    """
    # Check if the processed file exists
    for file in os.listdir(PROCESSED_DIR):
        if file.startswith(task_id):
            return {
                "status": "complete",
                "taskId": task_id,
                "audioUrl": f"/audio/{task_id}"
            }
    
    # Check if the file is still being processed
    for file in os.listdir(UPLOAD_DIR):
        if file.startswith(task_id):
            return {
                "status": "processing",
                "taskId": task_id
            }
    
    raise HTTPException(status_code=404, detail="Task not found")