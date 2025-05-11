#!/usr/bin/env python3

import argparse
import numpy as np
from pathlib import Path
import os
import secrets
import wave
import struct
from datetime import datetime

def parse_duration(duration_str):
    """
    Parse duration string in mm:ss format to seconds
    Time complexity: O(1)
    """
    try:
        minutes, seconds = map(int, duration_str.split(':'))
        return minutes * 60 + seconds
    except ValueError:
        raise ValueError("Duration must be in mm:ss format")

def generate_encryption_key(length):
    """
    Generate a random encryption key of specified length
    Time complexity: O(n) where n is the length of the key
    """
    return secrets.token_bytes(length)

def key_to_audio_samples(key, sample_rate=44100):
    """
    Convert encryption key bytes to audio samples
    Time complexity: O(n) where n is the length of the key
    """
    # Convert bytes to float values between -1 and 1
    samples = np.frombuffer(key, dtype=np.uint8)
    samples = (samples.astype(np.float32) / 128.0) - 1.0
    
    # Ensure samples are within [-1, 1] range
    samples = np.clip(samples, -1.0, 1.0)
    
    return samples

def generate_watermark(duration_seconds, sample_rate=44100):
    """
    Generate watermark audio samples for specified duration
    Time complexity: O(n) where n is the number of samples
    """
    # Calculate number of samples needed
    num_samples = int(duration_seconds * sample_rate)
    print(f"Generating {num_samples} samples for {duration_seconds} seconds at {sample_rate}Hz")
    
    # Generate encryption key with enough bytes to cover all samples
    key = generate_encryption_key(num_samples)
    print(f"Generated encryption key of length {len(key)} bytes")
    
    # Convert key to audio samples
    samples = key_to_audio_samples(key)
    print(f"Converted to {len(samples)} audio samples")
    
    return samples, sample_rate

def save_watermark(samples, sample_rate, output_dir="watermarks"):
    """
    Save watermark as WAV file
    Time complexity: O(n) where n is the number of samples
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"watermark_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)
    
    print(f"Saving {len(samples)} samples to {filepath}")
    
    # Convert float32 samples to int16
    samples_int16 = (samples * 32767).astype(np.int16)
    
    # Save as WAV file
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        
        # Write all samples at once
        wav_file.writeframes(samples_int16.tobytes())
    
    print(f"Successfully saved {len(samples)} samples")
    return filepath

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate audio watermark from encryption key')
    parser.add_argument('--duration', type=str, required=True,
                      help='Duration of watermark in mm:ss format')
    
    args = parser.parse_args()
    
    try:
        # Parse duration
        duration_seconds = parse_duration(args.duration)
        print(f"Requested duration: {args.duration} ({duration_seconds} seconds)")
        
        # Generate watermark
        print("Generating watermark...")
        samples, sample_rate = generate_watermark(duration_seconds)
        
        # Save watermark
        output_path = save_watermark(samples, sample_rate)
        print(f"Watermark saved to: {output_path}")
        print(f"Generated {len(samples)} samples at {sample_rate}Hz")
        print(f"Actual duration: {len(samples)/sample_rate:.2f} seconds")
        
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
