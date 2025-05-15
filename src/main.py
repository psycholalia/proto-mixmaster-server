import os
import uuid
from typing import List
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import librosa
import soundfile as sf
import numpy as np
import time

app = FastAPI(title="J Dilla Remix API")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for uploads and processed files
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Mount the processed directory to serve files
app.mount("/audio", StaticFiles(directory=PROCESSED_DIR), name="audio")

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
    # Load the audio file
    y, sr = librosa.load(input_path, sr=None)
    
    # Step 1: Beat detection
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Step 2: Apply swing feel (J Dilla's signature swing)
    # Apply more swing to even-numbered beats
    swung_beats = []
    for i, beat in enumerate(beat_times):
        if i % 2 == 1:  # Apply swing to every other beat
            swung_beats.append(beat + swing_amount * (beat_times[1] - beat_times[0]))
        else:
            swung_beats.append(beat)
    
    # Step 3: Time stretching for that "dragging" feel
    y_stretched = librosa.effects.time_stretch(y, rate=time_stretch_factor)
    
    # Step 4: Lo-fi effect
    # Add bit reduction effect
    bit_depth = 16 - int(10 * lofi_amount)  # Reduce bit depth for lo-fi effect
    y_quantized = np.round(y_stretched * (2**(bit_depth-1))) / (2**(bit_depth-1))
    
    # Add vinyl crackle
    crackle_amplitude = lofi_amount * 0.01
    vinyl_crackle = np.random.normal(0, crackle_amplitude, len(y_quantized))
    y_with_crackle = y_quantized + vinyl_crackle
    
    # Step 5: Slice and rearrange beats slightly (J Dilla often moved things off the grid)
    output_audio = np.zeros_like(y_with_crackle)
    beat_samples = librosa.time_to_samples(swung_beats, sr=sr)
    
    for i in range(len(beat_samples) - 1):
        start = beat_samples[i]
        end = beat_samples[i+1]
        
        if end >= len(y_with_crackle) or start >= len(y_with_crackle):
            continue
            
        # Get the current beat segment
        segment = y_with_crackle[start:end]
        
        # Random slight timing adjustments (quantize_strength controls how much we keep the original timing)
        if np.random.random() > 0.7:  # Only adjust some beats randomly
            adjustment = int(len(segment) * 0.08 * (1 - quantize_strength) * (np.random.random() - 0.5))
            start = max(0, start + adjustment)
            end = min(len(output_audio), end + adjustment)
        
        if end <= len(output_audio) and end > start:
            output_audio[start:end] = segment[:end-start]
    
    # Apply a subtle low-pass filter for warmth
    output_audio = librosa.effects.preemphasis(output_audio, coef=0.95)
    
    # Save the processed audio
    sf.write(output_path, output_audio, sr)
    
    return {"status": "complete", "file_path": output_path}

@app.post("/process-audio")
async def process_audio(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...)
):
    """
    Upload an MP3 file and process it to sound like J Dilla
    """
    # Validate that the file is an MP3
    if not audio.content_type or "audio/mpeg" not in audio.content_type:
        raise HTTPException(
            status_code=400, 
            detail="Only MP3 files are accepted"
        )
    
    # Generate a unique ID for this upload
    task_id = str(uuid.uuid4())
    file_extension = os.path.splitext(audio.filename)[1] if audio.filename else ".mp3"
    
    # Save the uploaded file
    input_path = os.path.join(UPLOAD_DIR, f"{task_id}{file_extension}")
    output_path = os.path.join(PROCESSED_DIR, f"{task_id}_dilla{file_extension}")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    # Process the audio in the background
    background_tasks.add_task(
        apply_j_dilla_effect,
        input_path=input_path,
        output_path=output_path,
        task_id=task_id
    )
    
    # For demo purposes, we'll add a slight delay to simulate processing
    # In a real application, you'd return a task ID and have a separate endpoint to check status
    time.sleep(2)  # Simulate processing time
    
    # Return the URL to the processed file
    return {
        "status": "success",
        "message": "Audio processed successfully",
        "audioUrl": f"/audio/{task_id}_dilla{file_extension}",
        "taskId": task_id
    }

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Check the status of a processing task
    """
    # In a real application, you would check the actual status from a database
    return {"status": "complete", "taskId": task_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)