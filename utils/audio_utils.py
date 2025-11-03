"""Audio processing utilities for AEC and VAD tasks."""

import torch
import numpy as np
from typing import Tuple, Optional


def apply_rir(signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """
    Apply Room Impulse Response (RIR) to a signal via convolution.
    
    Args:
        signal: Input signal, shape (n_samples,)
        rir: Room impulse response, shape (rir_length,)
    
    Returns:
        Convolved signal, same length as input signal
    """
    # Convolve signal with RIR
    convolved = np.convolve(signal, rir, mode='full')
    # Trim to original length
    return convolved[:len(signal)]


def mix_signals(clean: np.ndarray, 
                echo: np.ndarray, 
                noise: np.ndarray,
                ser_db: float = 0.0,
                snr_db: float = 10.0) -> np.ndarray:
    """
    Mix clean speech, echo, and noise with specified SER and SNR.
    
    Args:
        clean: Clean near-end speech signal
        echo: Echo signal
        noise: Noise signal
        ser_db: Signal-to-Echo Ratio in dB
        snr_db: Signal-to-Noise Ratio in dB
    
    Returns:
        Mixed microphone signal: y(t) = s(t) + e(t) + n(t)
    """
    # Ensure all signals have the same length
    min_len = min(len(clean), len(echo), len(noise))
    clean = clean[:min_len]
    echo = echo[:min_len]
    noise = noise[:min_len]
    
    # Calculate power
    clean_power = np.mean(clean ** 2)
    
    # Scale echo to achieve desired SER
    echo_power = np.mean(echo ** 2)
    if echo_power > 0:
        echo_scale = np.sqrt(clean_power / (echo_power * (10 ** (ser_db / 10))))
        echo = echo * echo_scale
    
    # Scale noise to achieve desired SNR
    noise_power = np.mean(noise ** 2)
    if noise_power > 0:
        noise_scale = np.sqrt(clean_power / (noise_power * (10 ** (snr_db / 10))))
        noise = noise * noise_scale
    
    # Mix signals
    mixed = clean + echo + noise
    
    # Normalize to prevent clipping
    max_val = np.abs(mixed).max()
    if max_val > 0.95:
        mixed = mixed * 0.95 / max_val
    
    return mixed


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio in dB.
    
    Args:
        signal: Clean signal
        noise: Noise signal
    
    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)


def stft_transform(signal: np.ndarray,
                   n_fft: int = 512,
                   hop_length: int = 128,
                   win_length: Optional[int] = None) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform.
    
    Args:
        signal: Input signal
        n_fft: FFT size
        hop_length: Hop length between frames
        win_length: Window length
    
    Returns:
        STFT coefficients, shape (n_freqs, n_frames)
    """
    if win_length is None:
        win_length = n_fft
    
    # Use Hann window
    window = np.hanning(win_length)
    
    # Pad signal
    pad_len = n_fft // 2
    signal = np.pad(signal, (pad_len, pad_len), mode='reflect')
    
    # Compute number of frames
    n_frames = 1 + (len(signal) - n_fft) // hop_length
    
    # Initialize STFT matrix
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    
    # Compute STFT
    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        
        # Apply window
        windowed = frame * window
        
        # Compute FFT
        spectrum = np.fft.rfft(windowed)
        stft_matrix[:, i] = spectrum
    
    return stft_matrix


def istft_transform(stft_matrix: np.ndarray,
                    hop_length: int = 128,
                    win_length: Optional[int] = None) -> np.ndarray:
    """
    Compute Inverse Short-Time Fourier Transform.
    
    Args:
        stft_matrix: STFT coefficients, shape (n_freqs, n_frames)
        hop_length: Hop length between frames
        win_length: Window length
    
    Returns:
        Reconstructed signal
    """
    n_freqs, n_frames = stft_matrix.shape
    n_fft = (n_freqs - 1) * 2
    
    if win_length is None:
        win_length = n_fft
    
    # Use Hann window
    window = np.hanning(win_length)
    
    # Initialize output signal
    signal_len = n_fft + (n_frames - 1) * hop_length
    signal = np.zeros(signal_len)
    window_sum = np.zeros(signal_len)
    
    # Overlap-add synthesis
    for i in range(n_frames):
        start = i * hop_length
        
        # Inverse FFT
        frame = np.fft.irfft(stft_matrix[:, i], n=n_fft)
        
        # Apply window and accumulate
        signal[start:start + n_fft] += frame * window
        window_sum[start:start + n_fft] += window ** 2
    
    # Normalize by window sum
    nonzero = window_sum > 1e-8
    signal[nonzero] /= window_sum[nonzero]
    
    # Remove padding
    pad_len = n_fft // 2
    signal = signal[pad_len:-pad_len]
    
    return signal


def generate_vad_labels(signal: np.ndarray,
                       sample_rate: int = 16000,
                       frame_length: float = 0.025,
                       hop_length: float = 0.010,
                       energy_threshold: float = 0.01) -> np.ndarray:
    """
    Generate frame-wise VAD labels based on energy threshold.
    
    Args:
        signal: Input audio signal
        sample_rate: Sample rate in Hz
        frame_length: Frame length in seconds
        hop_length: Hop length in seconds
        energy_threshold: Energy threshold for speech detection
    
    Returns:
        Binary VAD labels, shape (n_frames,)
    """
    frame_samples = int(frame_length * sample_rate)
    hop_samples = int(hop_length * sample_rate)
    
    n_frames = 1 + (len(signal) - frame_samples) // hop_samples
    labels = np.zeros(n_frames, dtype=np.int32)
    
    for i in range(n_frames):
        start = i * hop_samples
        frame = signal[start:start + frame_samples]
        
        # Compute frame energy
        energy = np.mean(frame ** 2)
        
        # Set label based on threshold
        labels[i] = 1 if energy > energy_threshold else 0
    
    return labels


def normalize_audio(signal: np.ndarray, target_level: float = -25.0) -> np.ndarray:
    """
    Normalize audio signal to target RMS level in dB.
    
    Args:
        signal: Input signal
        target_level: Target RMS level in dB
    
    Returns:
        Normalized signal
    """
    rms = np.sqrt(np.mean(signal ** 2))
    if rms == 0:
        return signal
    
    scalar = 10 ** (target_level / 20) / rms
    return signal * scalar
