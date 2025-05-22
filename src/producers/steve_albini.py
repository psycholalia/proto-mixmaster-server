import os
import librosa
import soundfile as sf
import numpy as np
import gc


# Constants for audio processing
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for file handling
MAX_AUDIO_LENGTH = 600  # Maximum audio length in seconds
SAMPLE_RATE = 44100  # Standard sample rate

async def apply_steve_albini_effect(
    input_path: str,
    output_path: str,
    task_id: str,
    dynamics_ratio: float = 0.8,
    noise_floor: float = 0.005,
    saturation: float = 0.3
):
    """
    Process audio to sound like Steve Albini's recording style:
    - Minimal compression (preserve dynamics)
    - Slight analog saturation
    - Natural room ambience
    - Raw, punchy character
    """
    try:
        # Load audio with high quality settings
        y, sr = librosa.load(
            input_path,
            sr=SAMPLE_RATE,
            duration=MAX_AUDIO_LENGTH,
            mono=True
        )

        # Process in smaller chunks to manage memory
        chunk_size = sr * 2  # 2-second chunks
        output = np.zeros_like(y)
        
        for i in range(0, len(y), chunk_size):
            chunk = y[i:min(i + chunk_size, len(y))]
            
            # Add subtle analog noise
            noise = np.random.normal(0, noise_floor, len(chunk))
            chunk = chunk + noise
            
            # Apply subtle tape saturation
            chunk = np.tanh(chunk * (1 + saturation)) / (1 + saturation)
            
            # Preserve dynamics (anti-compression)
            peaks = np.abs(chunk) > dynamics_ratio
            chunk[peaks] *= 1.2  # Enhance peaks
            
            # Enhance transients
            if len(chunk) > 1:
                envelope = np.abs(chunk)
                transients = np.diff(envelope, prepend=envelope[0]) > 0.1
                chunk[transients] *= 1.3
            
            # Store processed chunk
            chunk_end = min(i + chunk_size, len(output))
            output[i:chunk_end] = chunk[:chunk_end-i]
            
            # Force garbage collection
            del chunk
            gc.collect()

        # Add room ambience as a separate pass
        room_delay = int(sr * 0.02)  # 20ms room reflection
        if len(output) > room_delay:
            room = np.zeros_like(output)
            room[room_delay:] = output[:-room_delay] * 0.1
            output = output + room
            del room
            gc.collect()

        # Normalize while preserving dynamics
        max_amplitude = np.max(np.abs(output))
        if max_amplitude > 0:
            output = output / max_amplitude * 0.9

        # Save as MP3 format
        temp_wav = output_path + '.temp.wav'
        try:
            # First save as WAV
            sf.write(
                temp_wav,
                output,
                sr,
                format='WAV',
                subtype='PCM_16'
            )
            
            # Convert to MP3 using librosa and soundfile
            y_wav, sr_wav = librosa.load(temp_wav, sr=None)
            sf.write(
                output_path,
                y_wav,
                sr_wav,
                format='MP3'
            )
        finally:
            # Clean up temporary WAV file
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        del output
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
    
def process_in_chunks(y: np.ndarray, chunk_size: int, process_func):
    """Process audio data in chunks to reduce memory usage"""
    chunks = []
    for i in range(0, len(y), chunk_size):
        chunk = y[i:i + chunk_size]
        processed_chunk = process_func(chunk, i)  # Pass chunk index
        chunks.append(processed_chunk)
        gc.collect()  # Clean up after each chunk
    return np.concatenate(chunks)