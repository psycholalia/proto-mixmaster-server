import librosa
import soundfile as sf
import numpy as np

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
    
    return {"status": "complete", "file_path": output_path}