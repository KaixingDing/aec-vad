"""联合AEC-VAD数据预处理器

将AEC和VAD任务的数据整合到同一格式，生成可同时用于两个任务的训练样本。
"""

import os
import numpy as np
import soundfile as sf
from typing import Tuple, Optional, Dict
from pathlib import Path

from utils.audio_utils import apply_rir, mix_signals, normalize_audio, generate_vad_labels
from utils.scp_utils import write_scp


class JointAECVADPreprocessor:
    """
    联合AEC-VAD数据预处理器
    
    生成包含AEC和VAD所有必要信息的训练样本：
    - 麦克风信号（用于AEC和VAD）
    - 远端参考信号（用于AEC）
    - 近端纯净语音（AEC目标）
    - VAD标签（VAD目标）
    """
    
    def __init__(self,
                 dns_root: Optional[str] = None,
                 librispeech_root: Optional[str] = None,
                 sample_rate: int = 16000,
                 ser_range: Tuple[float, float] = (-5.0, 15.0),
                 snr_range: Tuple[float, float] = (0.0, 30.0),
                 duration: float = 4.0,
                 frame_length: float = 0.025,
                 hop_length: float = 0.010):
        """
        初始化联合预处理器
        
        Args:
            dns_root: DNS-Challenge数据集根目录
            librispeech_root: LibriSpeech数据集根目录
            sample_rate: 采样率（Hz）
            ser_range: 信号回声比范围（dB）
            snr_range: 信噪比范围（dB）
            duration: 样本时长（秒）
            frame_length: 帧长（秒，用于VAD标签）
            hop_length: 帧移（秒，用于VAD标签）
        """
        self.dns_root = Path(dns_root) if dns_root else None
        self.librispeech_root = Path(librispeech_root) if librispeech_root else None
        self.sample_rate = sample_rate
        self.ser_range = ser_range
        self.snr_range = snr_range
        self.duration = duration
        self.target_samples = int(duration * sample_rate)
        self.frame_length = frame_length
        self.hop_length = hop_length
        
        # 文件列表缓存
        self.clean_files = []
        self.rir_files = []
        self.noise_files = []
        self.speech_files = []
    
    def scan_files(self):
        """扫描并缓存音频文件路径"""
        # 扫描DNS-Challenge文件
        if self.dns_root and self.dns_root.exists():
            # 纯净语音
            for speech_type in ['read_speech', 'emotional_speech']:
                clean_dir = self.dns_root / 'clean' / speech_type
                if clean_dir.exists():
                    self.clean_files.extend(
                        list(clean_dir.rglob('*.wav')) + 
                        list(clean_dir.rglob('*.flac'))
                    )
            
            # 房间脉冲响应
            for rir_type in ['SLR26', 'SLR28']:
                rir_dir = self.dns_root / 'impulse_responses' / rir_type
                if rir_dir.exists():
                    self.rir_files.extend(
                        list(rir_dir.rglob('*.wav')) + 
                        list(rir_dir.rglob('*.flac'))
                    )
            
            # 噪声
            noise_dir = self.dns_root / 'noise'
            if noise_dir.exists():
                self.noise_files.extend(
                    list(noise_dir.rglob('*.wav')) + 
                    list(noise_dir.rglob('*.flac'))
                )
        
        # 扫描LibriSpeech文件
        if self.librispeech_root and self.librispeech_root.exists():
            for split in ['train-clean-100', 'train-clean-360', 'dev-clean']:
                split_dir = self.librispeech_root / split
                if split_dir.exists():
                    self.speech_files.extend(list(split_dir.rglob('*.flac')))
        
        # 如果没有LibriSpeech，使用DNS的纯净语音
        if not self.speech_files:
            self.speech_files = self.clean_files.copy()
        
        print(f"扫描到文件数量:")
        print(f"  纯净语音: {len(self.clean_files)}")
        print(f"  房间脉冲响应: {len(self.rir_files)}")
        print(f"  噪声: {len(self.noise_files)}")
        print(f"  语音（用于VAD）: {len(self.speech_files)}")
    
    def generate_joint_sample(self) -> Dict[str, np.ndarray]:
        """
        生成一个联合训练样本
        
        Returns:
            包含以下键的字典:
            - 'microphone': 麦克风信号 y(t) = s(t) + e(t) + n(t)
            - 'far_end': 远端参考信号 x(t)
            - 'near_end': 近端纯净语音 s(t)（AEC目标）
            - 'vad_labels': 逐帧VAD标签（VAD目标）
        """
        import random
        
        if not self.clean_files:
            raise ValueError("未找到文件，请先调用scan_files()")
        
        # 1. 选择近端语音（目标）
        near_end_file = random.choice(self.clean_files)
        near_end = self._load_audio(near_end_file)
        near_end = normalize_audio(near_end)
        
        # 2. 选择远端语音（不同于近端）
        far_end_file = random.choice(self.clean_files)
        while far_end_file == near_end_file and len(self.clean_files) > 1:
            far_end_file = random.choice(self.clean_files)
        far_end = self._load_audio(far_end_file)
        far_end = normalize_audio(far_end)
        
        # 3. 应用RIR生成回声
        if self.rir_files:
            rir_file = random.choice(self.rir_files)
            rir = self._load_audio(rir_file, target_length=None)
            rir = rir / (np.abs(rir).max() + 1e-8)
            echo = apply_rir(far_end, rir)
        else:
            echo = far_end * 0.5
        
        # 4. 选择噪声
        if self.noise_files:
            noise_file = random.choice(self.noise_files)
            noise = self._load_audio(noise_file)
            noise = normalize_audio(noise)
        else:
            noise = np.zeros_like(near_end)
        
        # 5. 混合信号
        ser_db = random.uniform(*self.ser_range)
        snr_db = random.uniform(*self.snr_range)
        microphone = mix_signals(near_end, echo, noise, ser_db=ser_db, snr_db=snr_db)
        
        # 6. 生成VAD标签
        # 基于近端语音的能量生成标签
        vad_labels = self._generate_vad_labels_from_signal(near_end)
        
        return {
            'microphone': microphone,
            'far_end': far_end,
            'near_end': near_end,
            'vad_labels': vad_labels,
        }
    
    def _load_audio(self, file_path: Path, target_length: Optional[int] = None) -> np.ndarray:
        """加载音频文件"""
        import random
        
        if target_length is None:
            target_length = self.target_samples
        
        signal, sr = sf.read(str(file_path))
        
        # 转换为单声道
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        
        # 重采样（简单线性插值）
        if sr != self.sample_rate:
            ratio = self.sample_rate / sr
            new_length = int(len(signal) * ratio)
            signal = np.interp(
                np.linspace(0, len(signal) - 1, new_length),
                np.arange(len(signal)),
                signal
            )
        
        # 调整长度
        if len(signal) < target_length:
            signal = np.pad(signal, (0, target_length - len(signal)))
        elif len(signal) > target_length:
            start = random.randint(0, len(signal) - target_length)
            signal = signal[start:start + target_length]
        
        return signal
    
    def _generate_vad_labels_from_signal(self, signal: np.ndarray) -> np.ndarray:
        """从信号生成VAD标签"""
        frame_samples = int(self.frame_length * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)
        
        n_frames = 1 + (len(signal) - frame_samples) // hop_samples
        labels = np.zeros(n_frames, dtype=np.int32)
        
        # 基于能量阈值
        energy_threshold = 0.01
        
        for i in range(n_frames):
            start = i * hop_samples
            frame = signal[start:start + frame_samples]
            energy = np.mean(frame ** 2)
            labels[i] = 1 if energy > energy_threshold else 0
        
        return labels
    
    def preprocess_dataset(self, 
                          num_samples: int,
                          output_dir: str,
                          split: str = 'train'):
        """
        预处理数据集并保存为SCP格式
        
        Args:
            num_samples: 生成的样本数量
            output_dir: 输出目录
            split: 数据集划分名称（'train', 'val', 'test'）
        """
        output_path = Path(output_dir) / split
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 准备SCP字典
        mic_scp = {}
        far_scp = {}
        near_scp = {}
        vad_scp = {}
        
        print(f"生成 {num_samples} 个 {split} 样本...")
        
        for i in range(num_samples):
            sample = self.generate_joint_sample()
            
            # 生成样本ID
            utt_id = f"{split}_{i:06d}"
            
            # 保存音频文件
            mic_path = output_path / 'microphone' / f'{utt_id}.wav'
            far_path = output_path / 'far_end' / f'{utt_id}.wav'
            near_path = output_path / 'near_end' / f'{utt_id}.wav'
            vad_path = output_path / 'vad_labels' / f'{utt_id}.npy'
            
            mic_path.parent.mkdir(parents=True, exist_ok=True)
            far_path.parent.mkdir(parents=True, exist_ok=True)
            near_path.parent.mkdir(parents=True, exist_ok=True)
            vad_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(mic_path, sample['microphone'], self.sample_rate)
            sf.write(far_path, sample['far_end'], self.sample_rate)
            sf.write(near_path, sample['near_end'], self.sample_rate)
            np.save(vad_path, sample['vad_labels'])
            
            # 添加到SCP字典
            mic_scp[utt_id] = str(mic_path.absolute())
            far_scp[utt_id] = str(far_path.absolute())
            near_scp[utt_id] = str(near_path.absolute())
            vad_scp[utt_id] = str(vad_path.absolute())
            
            if (i + 1) % 100 == 0:
                print(f"  已生成 {i + 1}/{num_samples} 样本")
        
        # 写入SCP文件
        write_scp(mic_scp, output_path / 'microphone.scp')
        write_scp(far_scp, output_path / 'far_end.scp')
        write_scp(near_scp, output_path / 'near_end.scp')
        write_scp(vad_scp, output_path / 'vad_labels.scp')
        
        print(f"完成 {split} 集生成")
        print(f"SCP文件保存在: {output_path}")
