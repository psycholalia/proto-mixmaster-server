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
import asyncio
from app import app
import gc
from contextlib import contextmanager

# Use Railway volume paths for storage

RAILWAY_VOLUME = os.getenv('RAILWAY_VOLUME_MOUNT_PATH', '/data')
UPLOAD_DIR = os.path.join(RAILWAY_VOLUME, "uploads")
PROCESSED_DIR = os.path.join(RAILWAY_VOLUME, "processed")

# Create directories in Railway volume
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Constants for audio processing
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for file handling
MAX_AUDIO_LENGTH = 600  # Maximum audio length in seconds
SAMPLE_RATE = 44100  # Standard sample rate

@contextmanager
def cleanup_files(*files):
    """Context manager to ensure file cleanup"""
    try:
        yield
    finally:
        for file in files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                print(f"Error cleaning up {file}: {e}")
        gc.collect()  # Force garbage collection

def process_in_chunks(y: np.ndarray, chunk_size: int, process_func):
    """Process audio data in chunks to reduce memory usage"""
    chunks = []
    for i in range(0, len(y), chunk_size):
        chunk = y[i:i + chunk_size]
        processed_chunk = process_func(chunk, i)  # Pass chunk index
        chunks.append(processed_chunk)
        gc.collect()  # Clean up after each chunk
    return np.concatenate(chunks)

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
    Process audio to sound like J Dilla style with optimized memory usage
    """
    try:
        # Load audio in chunks with memory cleanup
        y, sr = librosa.load(
            input_path, 
            sr=SAMPLE_RATE,
            duration=MAX_AUDIO_LENGTH,  # Limit maximum duration
            mono=True  # Force mono to reduce memory usage
        )

        # Early cleanup of input file
        if os.path.exists(input_path):
            os.remove(input_path)
        gc.collect()

        # Process beats in smaller chunks
        hop_length = 512
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, 
            sr=sr, 
            hop_length=hop_length,
            tightness=100  # Reduce computation complexity
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Apply swing in chunks
        def apply_swing(chunk, chunk_start_idx):
            chunk_copy = chunk.copy()
            start_time = chunk_start_idx / sr
            end_time = (chunk_start_idx + len(chunk)) / sr
            
            chunk_beats = np.where((beat_times >= start_time) & 
                                 (beat_times < end_time))[0]
            
            for i in chunk_beats:
                if i % 2 == 1 and i+1 < len(beat_times):
                    swing_samples = int(swing_amount * sr * (beat_times[1] - beat_times[0]))
                    beat_start = int((beat_times[i] - start_time) * sr)
                    if beat_start + swing_samples < len(chunk):
                        chunk_copy[beat_start:beat_start+swing_samples] = \
                            chunk[beat_start+swing_samples:beat_start+2*swing_samples]
            return chunk_copy

        # Process in chunks
        chunk_samples = sr * 5  # 5-second chunks
        y_swung = process_in_chunks(y, chunk_samples, apply_swing)

        # Time stretching in chunks
        def stretch_chunk(chunk, _):  # Ignore chunk index for stretching
            return librosa.effects.time_stretch(chunk, rate=time_stretch_factor)

        y_stretched = process_in_chunks(y_swung, chunk_samples, stretch_chunk)
        del y_swung
        gc.collect()

        # Lo-fi effects
        bit_depth = 16 - int(10 * lofi_amount)
        bit_crusher = float(2 ** (bit_depth - 1))
        
        def apply_lofi(chunk, _):  # Ignore chunk index for lo-fi effects
            # Quantize
            chunk_quantized = np.round(chunk * bit_crusher) / bit_crusher
            # Add vinyl noise (reduced amplitude)
            noise = np.random.normal(0, lofi_amount * 0.005, len(chunk_quantized))
            return chunk_quantized + noise

        y_processed = process_in_chunks(y_stretched, chunk_samples, apply_lofi)
        del y_stretched
        gc.collect()

        # Save with proper normalization
        max_amplitude = np.max(np.abs(y_processed))
        if max_amplitude > 0:
            y_processed = y_processed / max_amplitude * 0.95  # Prevent clipping

        # Write output file with proper cleanup
        sf.write(
            output_path, 
            y_processed, 
            sr,
            format='mp3',
            subtype='PCM_16'
        )
        
        del y_processed
        gc.collect()

        return {"status": "complete", "file_path": output_path}

    except Exception as e:
        print(f"Error processing audio: {e}")
        # Cleanup on error
        for path in [input_path, output_path]:
            if os.path.exists(path):
                os.remove(path)
        gc.collect()
        raise

@app.post("/process-audio")
async def process_audio(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...)
):
    """
    Upload and process audio file with proper resource management
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
    
    try:
        # Write file in chunks to manage memory
        with open(input_path, "wb") as buffer:
            while chunk := await audio.read(CHUNK_SIZE):
                buffer.write(chunk)
                time.sleep(0)  # Allow other tasks to run
        
        background_tasks.add_task(
            apply_j_dilla_effect,
            input_path=input_path,
            output_path=output_path,
            task_id=task_id
        )
        
        return {
            "status": "success",
            "message": "Audio processing started",
            "taskId": task_id
        }
    except Exception as e:
        # Cleanup on error
        for path in [input_path, output_path]:
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/audio/{file_id}")
async def get_audio(file_id: str):
    """
    Stream audio file response to manage memory
    """
    try:
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
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving audio: {str(e)}"
        )

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Check processing status
    """
    try:
        # Check processed file first
        for file in os.listdir(PROCESSED_DIR):
            if file.startswith(task_id):
                return {
                    "status": "complete",
                    "taskId": task_id,
                    "audioUrl": f"/audio/{task_id}"
                }
        
        # Check if still processing
        for file in os.listdir(UPLOAD_DIR):
            if file.startswith(task_id):
                return {
                    "status": "processing",
                    "taskId": task_id
                }
        
        raise HTTPException(status_code=404, detail="Task not found")
    except Exception as e:
            print('exception', e)
            raise HTTPException(
                status_code=500,
                detail=f"Error checking status: {str(e)}"
            )