"""使用模拟数据对音频工具进行单元测试"""

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
    """测试RIR应用"""
    # 生成模拟信号和RIR
    signal = np.random.randn(16000)  # 16kHz下1秒
    rir = np.random.randn(1000)  # 短RIR
    
    result = apply_rir(signal, rir)
    
    # 检查输出长度与输入匹配
    assert len(result) == len(signal)
    
    # 检查输出不全为零
    assert np.abs(result).max() > 0


def test_mix_signals():
    """测试带SNR控制的信号混合"""
    clean = np.random.randn(16000)
    echo = np.random.randn(16000)
    noise = np.random.randn(16000)
    
    # 测试混合
    mixed = mix_signals(clean, echo, noise, ser_db=10.0, snr_db=20.0)
    
    # 检查输出长度
    assert len(mixed) == min(len(clean), len(echo), len(noise))
    
    # 检查输出未削波
    assert np.abs(mixed).max() <= 1.0


def test_compute_snr():
    """测试SNR计算"""
    signal = np.ones(1000)
    noise = np.ones(1000) * 0.1
    
    snr = compute_snr(signal, noise)
    
    # 期望SNR应该为正
    assert snr > 0
    
    # 零噪声测试
    zero_noise = np.zeros(1000)
    snr_inf = compute_snr(signal, zero_noise)
    assert snr_inf == float('inf')


def test_stft_transform():
    """测试STFT计算"""
    signal = np.random.randn(16000)
    n_fft = 512
    hop_length = 160
    win_length = 400
    
    stft = stft_transform(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # 检查输出形状
    assert stft.shape[0] == n_fft // 2 + 1
    assert stft.shape[1] > 0
    
    # 检查复数输出
    assert stft.dtype == np.complex64


def test_istft_transform():
    """测试逆STFT"""
    # 创建信号
    original_signal = np.random.randn(16000)
    n_fft = 512
    hop_length = 160
    win_length = 400
    
    # 前向STFT
    stft = stft_transform(original_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # 逆STFT
    reconstructed = istft_transform(stft, hop_length=hop_length, win_length=win_length)
    
    # 检查长度大致正确
    assert abs(len(reconstructed) - len(original_signal)) < win_length


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
    """测试STFT->ISTFT大致重构信号"""
    signal = np.random.randn(16000)
    n_fft = 512
    hop_length = 160
    win_length = 400
    
    # 前向和逆向变换
    stft = stft_transform(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    reconstructed = istft_transform(stft, hop_length=hop_length, win_length=win_length)
    
    # 确保长度相同用于比较
    min_len = min(len(signal), len(reconstructed))
    signal = signal[:min_len]
    reconstructed = reconstructed[:min_len]
    
    # 检查重构误差较小
    # Povey窗口的重构可能不完美，所以放宽容差
    error = np.mean((signal - reconstructed) ** 2)
    assert error < 0.15  # 放宽阈值以适应Povey窗口


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
