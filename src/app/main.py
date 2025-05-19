import os
import uuid
from typing import List
import shutil
from fastapi import File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import librosa
import soundfile as sf
import numpy as np
import time
import tracemalloc
from app import app

# Use Railway volume paths for storage

RAILWAY_VOLUME = os.getenv('RAILWAY_VOLUME_MOUNT_PATH', 'server/data')
UPLOAD_DIR = os.path.join(RAILWAY_VOLUME, "uploads")
PROCESSED_DIR = os.path.join(RAILWAY_VOLUME, "processed")

# Create directories in Railway volume
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "J Dilla Remix API"}

async def apply_j_dilla_effect(
    input_path: str, 
    output_path: str,
    task_id: str,
    swing_amount: float = 0.3,
    quantize_strength: float = 0.7,
    time_stretch_factor: float = 0.98,
    lofi_amount: float = 0.4
):
    """
    Process audio to sound like J Dilla style with:
    - Beat slicing
    - Swing quantization
    - Time stretching
    - Lo-fi effects
    """
    tracemalloc.start()
    # Load the audio file
    y, sr = librosa.load(input_path, sr=None)
    
    # Step 1: Beat detection
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Step 2: Apply swing feel (J Dilla's signature swing)
    swung_beats = []
    for i, beat in enumerate(beat_times):
        if i % 2 == 1:  # Apply swing to every other beat
            swung_beats.append(beat + swing_amount * (beat_times[1] - beat_times[0]))
        else:
            swung_beats.append(beat)
    
    # Step 3: Time stretching for that "dragging" feel
    y_stretched = librosa.effects.time_stretch(y, rate=time_stretch_factor)
    
    # Step 4: Lo-fi effect
    bit_depth = 16 - int(10 * lofi_amount)  # Reduce bit depth for lo-fi effect
    y_quantized = np.round(y_stretched * (2**(bit_depth-1))) / (2**(bit_depth-1))
    
    # Add vinyl crackle
    crackle_amplitude = lofi_amount * 0.01
    vinyl_crackle = np.random.normal(0, crackle_amplitude, len(y_quantized))
    y_with_crackle = y_quantized + vinyl_crackle
    
    # Step 5: Slice and rearrange beats slightly
    output_audio = np.zeros_like(y_with_crackle)
    beat_samples = librosa.time_to_samples(swung_beats, sr=sr)
    
    for i in range(len(beat_samples) - 1):
        start = beat_samples[i]
        end = beat_samples[i+1]
        
        if end >= len(y_with_crackle) or start >= len(y_with_crackle):
            continue
            
        segment = y_with_crackle[start:end]
        
        if np.random.random() > 0.7:
            adjustment = int(len(segment) * 0.08 * (1 - quantize_strength) * (np.random.random() - 0.5))
            start = max(0, start + adjustment)
            end = min(len(output_audio), end + adjustment)
        
        if end <= len(output_audio) and end > start:
            output_audio[start:end] = segment[:end-start]
    
    # Apply a subtle low-pass filter for warmth
    output_audio = librosa.effects.preemphasis(output_audio, coef=0.95)
    gain = 15.5
    output_audio *= gain
    
    # Save the processed audio
    sf.write(output_path, output_audio, sr)
    print('trace mem', tracemalloc.get_traced_memory())
    tracemalloc.stop()
    
    return {"status": "complete", "file_path": output_path}

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
    output_path = os.path.join(PROCESSED_DIR, f"{task_id}_dilla{file_extension}")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    background_tasks.add_task(
        apply_j_dilla_effect,
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