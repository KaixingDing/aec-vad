"""音频处理工具：AEC和VAD任务专用

包含基于Povey窗口的STFT/iSTFT实现，与C端流式处理逻辑匹配。
"""

import torch
import numpy as np
from typing import Tuple, Optional


def povey_window(sr: int = 16000, 
                 win_size_ms: int = 25, 
                 n_fft_ms: int = 32, 
                 use_cuda: bool = False) -> torch.Tensor:
    """
    生成Povey窗口（基于Hann窗口的0.85次幂，并pad到与n_fft等长）
    
    Args:
        sr: 采样率（默认16000Hz）
        win_size_ms: 窗口大小（毫秒，默认25ms）
        n_fft_ms: FFT大小（毫秒，默认32ms）
        use_cuda: 是否使用CUDA
    
    Returns:
        Povey窗口张量，shape (n_fft,)
    """
    n_fft = int(sr * n_fft_ms / 1000)
    win_size = int(sr * win_size_ms / 1000)

    if n_fft < win_size:
        raise RuntimeError(f"错误 | POVEY_WINDOW | n_fft必须大于win_size, "
                           f"当前n_fft={n_fft}, win_size={win_size}")

    povey_window_ = list()

    M_2PI = 6.283185307179586476925286766559005
    a = M_2PI / (win_size - 1)

    for idx in range(win_size):
        povey_window_.append(pow(0.5 - 0.5 * np.cos(a * idx), 0.85))

    povey_window_ = np.array(povey_window_, dtype=np.float32)
    povey_window_ = np.concatenate([povey_window_, np.zeros(n_fft - win_size)], axis=-1)

    povey_window_ = torch.from_numpy(povey_window_)
    if use_cuda:
        povey_window_ = povey_window_.cuda()

    return povey_window_


def stft_transform(signal: np.ndarray,
                   n_fft: int = 512,
                   hop_length: int = 160,
                   win_length: int = 400,
                   sr: int = 16000) -> np.ndarray:
    """
    计算短时傅里叶变换（STFT），使用Povey窗口
    
    基于VOS SDK流式缓存逻辑实现，与C端代码匹配。
    
    Args:
        signal: 输入信号，shape (n_samples,)
        n_fft: FFT大小（默认512）
        hop_length: 帧移（默认160，对应10ms@16kHz）
        win_length: 窗口大小（默认400，对应25ms@16kHz）
        sr: 采样率（默认16000Hz）
    
    Returns:
        STFT系数，shape (n_freqs, n_frames)，复数类型
    """
    # 转换为torch张量
    if isinstance(signal, np.ndarray):
        signal_torch = torch.from_numpy(signal).unsqueeze(0)  # [1, T]
    else:
        signal_torch = signal.unsqueeze(0) if signal.dim() == 1 else signal
    
    # 生成Povey窗口
    n_fft_ms = int(n_fft * 1000 / sr)
    hop_size_ms = int(hop_length * 1000 / sr)
    win_size_ms = int(win_length * 1000 / sr)
    
    window = povey_window(sr, win_size_ms, n_fft_ms, use_cuda=signal_torch.is_cuda)
    
    # 检查信号长度
    if (signal_torch.size(1) - win_length) % hop_length != 0:
        # 填充使其满足条件
        target_len = win_length + ((signal_torch.size(1) - win_length) // hop_length) * hop_length
        if signal_torch.size(1) < target_len:
            pad_len = target_len - signal_torch.size(1)
            signal_torch = torch.nn.functional.pad(signal_torch, (0, pad_len))
        else:
            signal_torch = signal_torch[:, :target_len]
    
    n_cnt = (signal_torch.size(1) - win_length) // hop_length + 1

    chunk_list = list()
    for i in range(n_cnt):
        start_idx = i * hop_length
        end_idx = start_idx + win_length
        chunk_list.append(torch.unsqueeze(signal_torch[:, start_idx:end_idx], dim=0))

    # [n_cnt, B, win_size]
    chunk_list = torch.cat(chunk_list, dim=0)
    # [n_cnt*B, win_size]
    chunk_list = chunk_list.view(-1, win_length)

    # 在win_size后，尾端pad到与n_fft等长
    zero_pad = torch.zeros((chunk_list.shape[0], n_fft - win_length), device=chunk_list.device)
    # [n_cnt*B, n_fft]
    chunk_list_pad = torch.cat([chunk_list, zero_pad], dim=-1)

    # 进行STFT运算
    # [n_cnt*B, n_fft]
    stft_out = torch.fft.fft(chunk_list_pad * window)

    # [n_cnt*B, n_fft/2 + 1]
    out_half = stft_out[:, :n_fft // 2 + 1]
    # [n_cnt, B, n_fft/2 + 1]
    out_half = out_half.view((n_cnt, -1, n_fft // 2 + 1))
    # [B, n_cnt, n_fft/2 + 1]
    out_half = torch.transpose(out_half, 0, 1)
    
    # 转换回numpy格式 (n_freqs, n_frames)
    real_part = out_half.real.float().squeeze(0).transpose(0, 1).numpy()  # [n_fft/2+1, n_cnt]
    imag_part = out_half.imag.float().squeeze(0).transpose(0, 1).numpy()
    
    # 组合为复数
    stft_matrix = real_part + 1j * imag_part
    
    return stft_matrix.astype(np.complex64)


def istft_transform(stft_matrix: np.ndarray,
                    hop_length: int = 160,
                    win_length: int = 400,
                    sr: int = 16000) -> np.ndarray:
    """
    计算逆短时傅里叶变换（iSTFT），使用Povey窗口
    
    基于VOS SDK流式缓存逻辑实现，与C端代码匹配。
    
    Args:
        stft_matrix: STFT系数，shape (n_freqs, n_frames)
        hop_length: 帧移（默认160，对应10ms@16kHz）
        win_length: 窗口大小（默认400，对应25ms@16kHz）
        sr: 采样率（默认16000Hz）
    
    Returns:
        重构信号，shape (n_samples,)
    """
    n_freqs, n_frames = stft_matrix.shape
    n_fft = (n_freqs - 1) * 2
    
    # 分离实部和虚部，转换为torch张量
    real_part = torch.from_numpy(np.real(stft_matrix)).transpose(0, 1).unsqueeze(0)  # [1, n_frames, n_freqs]
    imag_part = torch.from_numpy(np.imag(stft_matrix)).transpose(0, 1).unsqueeze(0)
    
    # 生成Povey窗口
    n_fft_ms = int(n_fft * 1000 / sr)
    hop_size_ms = int(hop_length * 1000 / sr)
    win_size_ms = int(win_length * 1000 / sr)
    
    window = povey_window(sr, win_size_ms, n_fft_ms, use_cuda=real_part.is_cuda)
    
    # 合并成复数
    complex_input = torch.complex(real_part, imag_part)

    # 进行iFFT运算
    # [B, n_frames, n_fft]
    istft_out = torch.fft.irfft(complex_input, dim=-1)

    # 将窗函数作用回ifft的结果上
    istft_out *= window
    # [B, n_frames, win_size]
    istft_out = istft_out[:, :, :win_length]

    # 设置输出音频的shape
    B = istft_out.size(0)
    n_frames_actual = istft_out.size(1)

    length = (n_frames_actual - 1) * hop_length + win_length
    final_wav = torch.zeros(B, length, device=istft_out.device)

    # 对每个win_size进行overlap-add
    for i in range(n_frames_actual):
        start_idx = i * hop_length
        end_idx = start_idx + win_length
        # [B, win_length]
        cur_istft = istft_out[:, i, :]
        # 进行overlap-add
        final_wav[:, start_idx:end_idx] += cur_istft

    # 用窗函数平方和进行归一化
    ifft_window_sum = _window_sumsquare(window, n_frames_actual, False, win_length, hop_length, n_fft)
    # 移除极小值，这样只会影响音频头部和尾部的10ms无法完美还原
    approx_nonzero_indices = ifft_window_sum > 0.9

    final_wav[:, approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    return final_wav.squeeze(0).numpy()


def _window_sumsquare(window: torch.Tensor, 
                     n_frames: int, 
                     use_cuda: bool, 
                     win_size: int = 400, 
                     hop_length: int = 160, 
                     n_fft: int = 512) -> torch.Tensor:
    """
    对pad后的窗函数进行平方滑动加和
    
    Args:
        window: 窗函数（已pad到n_fft长度）
        n_frames: 帧数
        use_cuda: 是否使用CUDA
        win_size: 窗口大小
        hop_length: 帧移
        n_fft: FFT大小
    
    Returns:
        窗函数平方和，shape (n,)
    """
    assert window.size(0) == n_fft

    n = win_size + hop_length * (n_frames - 1)
    x = torch.zeros(n, device=window.device)

    win_sq = window ** 2

    for i in range(n_frames):
        start_idx = i * hop_length
        x[start_idx:min(n, start_idx + n_fft)] += win_sq[:max(0, min(n_fft, n - start_idx))]

    return x


def apply_rir(signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """
    通过卷积将房间冲激响应(RIR)应用到信号上
    
    Args:
        signal: 输入信号，shape (n_samples,)
        rir: 房间冲激响应，shape (rir_length,)
    
    Returns:
        卷积后的信号，与输入信号长度相同
    """
    # 与RIR进行卷积
    convolved = np.convolve(signal, rir, mode='full')
    # 截断到原始长度
    return convolved[:len(signal)]


def mix_signals(clean: np.ndarray, 
                echo: np.ndarray, 
                noise: np.ndarray,
                ser_db: float = 0.0,
                snr_db: float = 10.0) -> np.ndarray:
    """
    混合纯净语音、回声和噪声，指定SER和SNR
    
    Args:
        clean: 纯净近端语音信号
        echo: 回声信号
        noise: 噪声信号
        ser_db: 信号回声比（dB）
        snr_db: 信噪比（dB）
    
    Returns:
        混合麦克风信号: y(t) = s(t) + e(t) + n(t)
    """
    # 确保所有信号长度相同
    min_len = min(len(clean), len(echo), len(noise))
    clean = clean[:min_len]
    echo = echo[:min_len]
    noise = noise[:min_len]
    
    # 计算功率
    clean_power = np.mean(clean ** 2)
    
    # 缩放回声以达到目标SER
    echo_power = np.mean(echo ** 2)
    if echo_power > 0:
        echo_scale = np.sqrt(clean_power / (echo_power * (10 ** (ser_db / 10))))
        echo = echo * echo_scale
    
    # 缩放噪声以达到目标SNR
    noise_power = np.mean(noise ** 2)
    if noise_power > 0:
        noise_scale = np.sqrt(clean_power / (noise_power * (10 ** (snr_db / 10))))
        noise = noise * noise_scale
    
    # 混合信号
    mixed = clean + echo + noise
    
    # 归一化防止削波
    max_val = np.abs(mixed).max()
    if max_val > 0.95:
        mixed = mixed * 0.95 / max_val
    
    return mixed


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    计算信噪比（SNR），单位dB
    
    Args:
        signal: 纯净信号
        noise: 噪声信号
    
    Returns:
        SNR（dB）
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)


def generate_vad_labels(signal: np.ndarray,
                       sample_rate: int = 16000,
                       frame_length: float = 0.025,
                       hop_length: float = 0.010,
                       energy_threshold: float = 0.01) -> np.ndarray:
    """
    基于能量阈值生成逐帧VAD标签
    
    Args:
        signal: 输入音频信号
        sample_rate: 采样率（Hz）
        frame_length: 帧长度（秒）
        hop_length: 帧移（秒）
        energy_threshold: 语音检测的能量阈值
    
    Returns:
        二值VAD标签，shape (n_frames,)
    """
    frame_samples = int(frame_length * sample_rate)
    hop_samples = int(hop_length * sample_rate)
    
    n_frames = 1 + (len(signal) - frame_samples) // hop_samples
    labels = np.zeros(n_frames, dtype=np.int32)
    
    for i in range(n_frames):
        start = i * hop_samples
        frame = signal[start:start + frame_samples]
        
        # 计算帧能量
        energy = np.mean(frame ** 2)
        
        # 基于阈值设置标签
        labels[i] = 1 if energy > energy_threshold else 0
    
    return labels


def normalize_audio(signal: np.ndarray, target_level: float = -25.0) -> np.ndarray:
    """
    将音频信号归一化到目标RMS电平（dB）
    
    Args:
        signal: 输入信号
        target_level: 目标RMS电平（dB）
    
    Returns:
        归一化后的信号
    """
    rms = np.sqrt(np.mean(signal ** 2))
    if rms == 0:
        return signal
    
    scalar = 10 ** (target_level / 20) / rms
    return signal * scalar


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
