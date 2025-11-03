"""Unit tests for audio utilities using mock data."""

import pytest
import numpy as np
from utils.audio_utils import (
    apply_rir,
    mix_signals,
    compute_snr,
    stft_transform,
    istft_transform,
    generate_vad_labels,
    normalize_audio,
)


def test_apply_rir():
    """Test RIR application."""
    # Generate mock signal and RIR
    signal = np.random.randn(16000)  # 1 second at 16kHz
    rir = np.random.randn(1000)  # Short RIR
    
    result = apply_rir(signal, rir)
    
    # Check output length matches input
    assert len(result) == len(signal)
    
    # Check output is not all zeros
    assert np.abs(result).max() > 0


def test_mix_signals():
    """Test signal mixing with SNR control."""
    clean = np.random.randn(16000)
    echo = np.random.randn(16000)
    noise = np.random.randn(16000)
    
    # Test mixing
    mixed = mix_signals(clean, echo, noise, ser_db=10.0, snr_db=20.0)
    
    # Check output length
    assert len(mixed) == min(len(clean), len(echo), len(noise))
    
    # Check output is not clipping
    assert np.abs(mixed).max() <= 1.0


def test_compute_snr():
    """Test SNR computation."""
    signal = np.ones(1000)
    noise = np.ones(1000) * 0.1
    
    snr = compute_snr(signal, noise)
    
    # Expected SNR should be positive
    assert snr > 0
    
    # Test with zero noise
    zero_noise = np.zeros(1000)
    snr_inf = compute_snr(signal, zero_noise)
    assert snr_inf == float('inf')


def test_stft_transform():
    """Test STFT computation."""
    signal = np.random.randn(16000)
    n_fft = 512
    hop_length = 128
    
    stft = stft_transform(signal, n_fft=n_fft, hop_length=hop_length)
    
    # Check output shape
    assert stft.shape[0] == n_fft // 2 + 1
    assert stft.shape[1] > 0
    
    # Check complex output
    assert stft.dtype == np.complex64


def test_istft_transform():
    """Test inverse STFT."""
    # Create a signal
    original_signal = np.random.randn(16000)
    n_fft = 512
    hop_length = 128
    
    # Forward STFT
    stft = stft_transform(original_signal, n_fft=n_fft, hop_length=hop_length)
    
    # Inverse STFT
    reconstructed = istft_transform(stft, hop_length=hop_length)
    
    # Check length is approximately correct
    assert abs(len(reconstructed) - len(original_signal)) < n_fft


def test_generate_vad_labels():
    """Test VAD label generation."""
    # Create signal with clear speech (high energy) and silence (low energy)
    sample_rate = 16000
    duration = 2.0
    
    # High energy segment (speech)
    speech = np.random.randn(int(sample_rate * 1.0)) * 0.5
    
    # Low energy segment (silence)
    silence = np.random.randn(int(sample_rate * 1.0)) * 0.01
    
    # Concatenate
    signal = np.concatenate([speech, silence])
    
    # Generate labels
    labels = generate_vad_labels(signal, sample_rate=sample_rate)
    
    # Check output shape
    assert len(labels) > 0
    
    # Check labels are binary
    assert set(labels).issubset({0, 1})


def test_normalize_audio():
    """Test audio normalization."""
    signal = np.random.randn(16000) * 10  # High amplitude
    
    normalized = normalize_audio(signal, target_level=-25.0)
    
    # Check RMS is close to target
    rms = np.sqrt(np.mean(normalized ** 2))
    target_rms = 10 ** (-25.0 / 20)
    
    assert abs(rms - target_rms) < 0.01
    
    # Test with zero signal
    zero_signal = np.zeros(1000)
    normalized_zero = normalize_audio(zero_signal)
    assert np.all(normalized_zero == 0)


def test_stft_istft_reconstruction():
    """Test that STFT->ISTFT approximately reconstructs signal."""
    signal = np.random.randn(16000)
    n_fft = 512
    hop_length = 128
    
    # Forward and backward
    stft = stft_transform(signal, n_fft=n_fft, hop_length=hop_length)
    reconstructed = istft_transform(stft, hop_length=hop_length)
    
    # Ensure same length for comparison
    min_len = min(len(signal), len(reconstructed))
    signal = signal[:min_len]
    reconstructed = reconstructed[:min_len]
    
    # Check reconstruction error is small
    error = np.mean((signal - reconstructed) ** 2)
    assert error < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
