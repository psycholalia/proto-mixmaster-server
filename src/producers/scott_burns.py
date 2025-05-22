import os
import librosa
import soundfile as sf
import numpy as np
import gc


# Constants for audio processing
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for file handling
MAX_AUDIO_LENGTH = 600  # Maximum audio length in seconds
SAMPLE_RATE = 44100  # Standard sample rate

async def apply_scott_burns_effect(
    input_path: str,
    output_path: str,
    task_id: str,
    bass_boost: float = 0.4,
    high_end_crisp: float = 0.3,
    drum_punch: float = 0.5,
    distortion: float = 0.2
):
    """
    Process audio to sound like Scott Burns' death metal production style:
    - Enhanced low-end punch
    - Crisp high frequencies
    - Aggressive midrange
    - Tight drum processing
    - Controlled distortion
    """
    try:
        # Load audio with high quality settings
        y, sr = librosa.load(
            input_path,
            sr=SAMPLE_RATE,
            duration=MAX_AUDIO_LENGTH,
            mono=True
        )

        # Early cleanup
        if os.path.exists(input_path):
            os.remove(input_path)
        gc.collect()

        # Process in chunks
        chunk_size = sr * 2  # 2-second chunks
        output = np.zeros_like(y)
        
        def process_chunk(chunk, _):
            # Enhance low end (typical Scott Burns bass treatment)
            bass_frequencies = librosa.effects.preemphasis(chunk, coef=-0.95)
            bass_enhanced = bass_frequencies * (1 + bass_boost)
            
            # Add high-end crispness
            high_frequencies = librosa.effects.preemphasis(chunk, coef=0.95)
            high_enhanced = high_frequencies * (1 + high_end_crisp)
            
            # Combine frequencies with proper balance
            processed = (bass_enhanced * 0.6 + high_enhanced * 0.4)
            
            # Add controlled distortion (characteristic of death metal production)
            processed = np.tanh(processed * (1 + distortion))
            
            # Enhance transients for drum punch
            if len(processed) > 1:
                transients = np.diff(np.abs(processed), prepend=processed[0])
                transient_mask = transients > np.mean(np.abs(transients))
                processed[transient_mask] *= (1 + drum_punch)
            
            return processed

        # Process audio in chunks
        output = process_in_chunks(y, chunk_size, process_chunk)
        del y
        gc.collect()

        # Final processing stage
        def apply_final_processing(chunk, _):
            # Add subtle harmonics (characteristic of analog gear)
            harmonics = np.sin(2 * np.pi * np.arange(len(chunk)) * 2 / sr) * 0.02
            chunk = chunk + harmonics
            
            # Aggressive but controlled compression
            threshold = 0.3
            ratio = 4.0
            chunk_compressed = np.where(
                np.abs(chunk) > threshold,
                threshold + (np.abs(chunk) - threshold) / ratio * np.sign(chunk),
                chunk
            )
            
            return chunk_compressed

        output = process_in_chunks(output, chunk_size, apply_final_processing)

        # Normalize while preserving some dynamics (Scott Burns style)
        max_amplitude = np.max(np.abs(output))
        if max_amplitude > 0:
            output = output / max_amplitude * 0.95

        # Save as MP3 with appropriate quality settings
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