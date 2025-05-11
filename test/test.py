#!/usr/bin/env python3

import argparse
import numpy as np
import wave
import os
from pathlib import Path
import matplotlib.pyplot as plt

def load_audio(filepath):
    """
    Load audio file and return samples and sample rate
    Time complexity: O(n) where n is the number of samples
    """
    with wave.open(filepath, 'rb') as wav_file:
        # Get audio parameters
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        # Read audio data
        frames = wav_file.readframes(n_frames)
        
        # Convert to numpy array
        if sample_width == 2:  # 16-bit
            dtype = np.int16
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        samples = np.frombuffer(frames, dtype=dtype)
        
        # Convert to float32 and normalize
        samples = samples.astype(np.float32) / 32767.0
        
        # Convert to mono if stereo
        if n_channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        
        return samples, sample_rate

def perform_null_test(original, watermarked):
    """
    Perform null test between original and watermarked audio
    Time complexity: O(n) where n is the number of samples
    """
    # Ensure both files are the same length
    min_length = min(len(original), len(watermarked))
    original = original[:min_length]
    watermarked = watermarked[:min_length]
    
    # Invert original and mix with watermarked
    null_result = watermarked - original
    
    return null_result

def compare_with_original_watermark(null_result, original_watermark, sample_rate):
    """
    Compare null test result with original watermark
    Time complexity: O(n log n) where n is the number of samples
    """
    # Ensure both files are the same length
    min_length = min(len(null_result), len(original_watermark))
    null_result = null_result[:min_length]
    original_watermark = original_watermark[:min_length]
    
    # Use FFT-based correlation for better performance
    # Pad to next power of 2 for efficient FFT
    n = 2**np.ceil(np.log2(len(null_result) + len(original_watermark) - 1)).astype(int)
    
    # Compute FFTs
    fft_null = np.fft.rfft(null_result, n)
    fft_wm = np.fft.rfft(original_watermark, n)
    
    # Compute correlation in frequency domain
    correlation = np.fft.irfft(fft_null * np.conj(fft_wm))
    
    # Normalize
    max_correlation = np.max(np.abs(correlation))
    if max_correlation > 0:
        correlation = correlation / max_correlation
    
    # Find the peak correlation and its offset
    peak_idx = np.argmax(np.abs(correlation))
    peak_offset = peak_idx - (len(null_result) - 1)
    
    return correlation, peak_offset, max_correlation

def analyze_null_result(null_result, sample_rate, original_watermark=None):
    """
    Analyze null test result to detect watermark
    Time complexity: O(n log n) where n is the number of samples
    """
    print("Calculating basic statistics...")
    # Calculate statistics
    rms = np.sqrt(np.mean(null_result**2))
    peak = np.max(np.abs(null_result))
    dc_offset = np.mean(null_result)
    
    print("Computing frequency spectrum...")
    # Calculate frequency spectrum
    spectrum = np.abs(np.fft.rfft(null_result))
    freq = np.fft.rfftfreq(len(null_result), 1/sample_rate)
    
    # Calculate energy in different frequency bands
    def band_energy(low, high):
        mask = (freq >= low) & (freq <= high)
        return np.sum(spectrum[mask])
    
    # Define frequency bands (Hz)
    bands = {
        'sub': (20, 60),
        'low': (60, 250),
        'mid': (250, 2000),
        'high': (2000, 20000)
    }
    
    print("Analyzing frequency bands...")
    band_energies = {name: band_energy(low, high) 
                    for name, (low, high) in bands.items()}
    
    # Calculate confidence score
    # Higher RMS and more even distribution across bands suggests watermark
    total_energy = sum(band_energies.values())
    if total_energy == 0:
        return 0.0, band_energies, None
    
    # Normalize band energies
    normalized_bands = {name: energy/total_energy 
                       for name, energy in band_energies.items()}
    
    # Calculate evenness of distribution (higher is better)
    evenness = 1 - np.std(list(normalized_bands.values()))
    
    # Calculate base confidence score
    confidence = rms * evenness * (1 - abs(dc_offset))
    
    # If original watermark is provided, use it to adjust confidence
    correlation_info = None
    if original_watermark is not None:
        print("Computing correlation with original watermark...")
        correlation, offset, max_corr = compare_with_original_watermark(
            null_result, original_watermark, sample_rate)
        
        # Adjust confidence based on correlation
        correlation_factor = max_corr / (rms * len(null_result))
        confidence *= (1 + correlation_factor)
        
        correlation_info = {
            'correlation': correlation,
            'offset': offset,
            'max_correlation': max_corr
        }
    
    return confidence, band_energies, correlation_info

def plot_analysis(null_result, sample_rate, band_energies, confidence, correlation_info=None):
    """
    Create visualization of the analysis
    Time complexity: O(n) where n is the number of samples
    """
    # Create figure with subplots
    n_plots = 3 if correlation_info else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4*n_plots))
    
    # Plot waveform
    time = np.arange(len(null_result)) / sample_rate
    axes[0].plot(time, null_result)
    axes[0].set_title('Null Test Result Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    
    # Plot frequency bands
    bands = list(band_energies.keys())
    energies = list(band_energies.values())
    axes[1].bar(bands, energies)
    axes[1].set_title('Energy Distribution Across Frequency Bands')
    axes[1].set_xlabel('Frequency Band')
    axes[1].set_ylabel('Energy')
    
    # Plot correlation if available
    if correlation_info:
        correlation = correlation_info['correlation']
        offset = correlation_info['offset']
        max_corr = correlation_info['max_correlation']
        
        axes[2].plot(correlation)
        axes[2].axvline(x=len(null_result)-1, color='r', linestyle='--', 
                       label='Zero offset')
        axes[2].axvline(x=len(null_result)-1+offset, color='g', linestyle='--',
                       label=f'Peak offset: {offset} samples')
        axes[2].set_title(f'Correlation with Original Watermark\nMax Correlation: {max_corr:.3f}')
        axes[2].set_xlabel('Offset (samples)')
        axes[2].set_ylabel('Correlation')
        axes[2].legend()
    
    # Add confidence score to plot
    plt.suptitle(f'Watermark Detection Analysis\nConfidence Score: {confidence:.3f}')
    
    # Save plot
    output_dir = Path('test/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'watermark_analysis.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Test for watermark presence in audio')
    parser.add_argument('--original', type=str, required=True,
                      help='Path to original audio file')
    parser.add_argument('--watermarked', type=str, required=True,
                      help='Path to watermarked audio file')
    parser.add_argument('--watermark', type=str,
                      help='Path to original watermark file (optional)')
    parser.add_argument('--plot', action='store_true',
                      help='Generate analysis plots')
    
    args = parser.parse_args()
    
    try:
        # Load audio files
        print(f"Loading original audio: {args.original}")
        original, orig_sr = load_audio(args.original)
        
        print(f"Loading watermarked audio: {args.watermarked}")
        watermarked, wm_sr = load_audio(args.watermarked)
        
        # Load original watermark if provided
        original_watermark = None
        if args.watermark:
            print(f"Loading original watermark: {args.watermark}")
            original_watermark, wm_sr = load_audio(args.watermark)
        
        # Verify sample rates match
        if orig_sr != wm_sr:
            raise ValueError(f"Sample rates don't match: {orig_sr}Hz vs {wm_sr}Hz")
        
        # Perform null test
        print("Performing null test...")
        null_result = perform_null_test(original, watermarked)
        
        # Analyze result
        print("Analyzing results...")
        confidence, band_energies, correlation_info = analyze_null_result(
            null_result, orig_sr, original_watermark)
        
        # Print results
        print("\nAnalysis Results:")
        print(f"Confidence Score: {confidence:.3f}")
        print("\nEnergy Distribution:")
        for band, energy in band_energies.items():
            print(f"{band}: {energy:.3f}")
        
        if correlation_info:
            print("\nCorrelation with Original Watermark:")
            print(f"Maximum Correlation: {correlation_info['max_correlation']:.3f}")
            print(f"Offset: {correlation_info['offset']} samples")
        
        # Generate plots if requested
        if args.plot:
            print("\nGenerating analysis plots...")
            plot_analysis(null_result, orig_sr, band_energies, confidence, correlation_info)
            print(f"Plots saved to test/results/watermark_analysis.png")
        
        # Determine if watermark is present
        threshold = 0.1  # Adjust this threshold based on testing
        if confidence > threshold:
            print(f"\nWatermark detected with {confidence:.1%} confidence")
        else:
            print(f"\nNo watermark detected (confidence: {confidence:.1%})")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
