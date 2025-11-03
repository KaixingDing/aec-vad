"""基于SCP文件的联合AEC-VAD数据集

从SCP文件加载预处理好的数据，同时支持AEC和VAD任务训练。
"""

import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, List
from pathlib import Path

from utils.scp_utils import read_scp
from utils.audio_utils import stft_transform


class SCPDataset(Dataset):
    """
    基于SCP文件的联合AEC-VAD数据集
    
    从SCP文件读取预处理好的数据，支持同时进行AEC和VAD任务训练。
    每个样本包含：
    - 麦克风信号
    - 远端参考信号
    - 近端纯净语音（AEC目标）
    - VAD标签（VAD目标）
    """
    
    def __init__(self,
                 microphone_scp: str,
                 far_end_scp: str,
                 near_end_scp: str,
                 vad_labels_scp: str,
                 n_fft: int = 512,
                 hop_length: int = 128,
                 sample_rate: int = 16000):
        """
        初始化SCP数据集
        
        Args:
            microphone_scp: 麦克风信号SCP文件路径
            far_end_scp: 远端信号SCP文件路径
            near_end_scp: 近端纯净语音SCP文件路径
            vad_labels_scp: VAD标签SCP文件路径
            n_fft: STFT的FFT大小
            hop_length: STFT的帧移
            sample_rate: 采样率
        """
        # 读取SCP文件
        self.microphone_dict = read_scp(microphone_scp)
        self.far_end_dict = read_scp(far_end_scp)
        self.near_end_dict = read_scp(near_end_scp)
        self.vad_labels_dict = read_scp(vad_labels_scp)
        
        # 获取样本ID列表（使用所有SCP文件的交集）
        self.utt_ids = sorted(
            set(self.microphone_dict.keys()) &
            set(self.far_end_dict.keys()) &
            set(self.near_end_dict.keys()) &
            set(self.vad_labels_dict.keys())
        )
        
        if not self.utt_ids:
            raise ValueError("未找到有效样本。请检查SCP文件是否包含相同的样本ID。")
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        print(f"加载了 {len(self.utt_ids)} 个样本")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.utt_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个训练样本
        
        Args:
            idx: 样本索引
        
        Returns:
            包含AEC和VAD所需数据的字典
        """
        utt_id = self.utt_ids[idx]
        
        # 加载音频文件
        microphone, _ = sf.read(self.microphone_dict[utt_id])
        far_end, _ = sf.read(self.far_end_dict[utt_id])
        near_end, _ = sf.read(self.near_end_dict[utt_id])
        
        # 加载VAD标签
        vad_labels = np.load(self.vad_labels_dict[utt_id])
        
        # 计算STFT
        mic_stft = stft_transform(microphone, n_fft=self.n_fft, hop_length=self.hop_length)
        far_end_stft = stft_transform(far_end, n_fft=self.n_fft, hop_length=self.hop_length)
        near_end_stft = stft_transform(near_end, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # 提取幅度和相位
        mic_mag = np.abs(mic_stft)
        mic_phase = np.angle(mic_stft)
        far_end_mag = np.abs(far_end_stft)
        near_end_mag = np.abs(near_end_stft)
        near_end_phase = np.angle(near_end_stft)
        
        # AEC输入：堆叠麦克风和远端信号的幅度（2通道）
        aec_input = np.stack([mic_mag, far_end_mag], axis=0)  # (2, n_freqs, n_frames)
        
        # VAD输入：仅使用麦克风信号的幅度（1通道）
        vad_input = mic_mag[np.newaxis, :, :]  # (1, n_freqs, n_frames)
        
        # 确保VAD标签与帧数匹配
        n_frames = mic_mag.shape[1]
        if len(vad_labels) > n_frames:
            vad_labels = vad_labels[:n_frames]
        elif len(vad_labels) < n_frames:
            vad_labels = np.pad(vad_labels, (0, n_frames - len(vad_labels)))
        
        # 转换为torch张量
        return {
            'utt_id': utt_id,
            # AEC任务数据
            'aec_input': torch.FloatTensor(aec_input),
            'aec_target_mag': torch.FloatTensor(near_end_mag),
            'aec_target_phase': torch.FloatTensor(near_end_phase),
            'mic_phase': torch.FloatTensor(mic_phase),
            # VAD任务数据
            'vad_input': torch.FloatTensor(vad_input),
            'vad_labels': torch.LongTensor(vad_labels),
        }


def collate_scp_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    SCP数据集的批处理整理函数
    
    将多个样本整理成批次，自动处理AEC和VAD任务的数据。
    
    Args:
        batch: 样本列表
    
    Returns:
        整理后的批次字典
    """
    # 收集所有样本ID
    utt_ids = [sample['utt_id'] for sample in batch]
    
    # 整理AEC数据
    aec_inputs = torch.stack([sample['aec_input'] for sample in batch])
    aec_target_mags = torch.stack([sample['aec_target_mag'] for sample in batch])
    aec_target_phases = torch.stack([sample['aec_target_phase'] for sample in batch])
    mic_phases = torch.stack([sample['mic_phase'] for sample in batch])
    
    # 整理VAD数据（需要处理可能的长度差异）
    max_frames = max(sample['vad_labels'].shape[0] for sample in batch)
    
    vad_inputs = []
    vad_labels = []
    
    for sample in batch:
        vad_input = sample['vad_input']
        vad_label = sample['vad_labels']
        
        # 填充到最大帧数
        n_frames = vad_input.shape[2]
        if n_frames < max_frames:
            pad_size = max_frames - n_frames
            vad_input = torch.nn.functional.pad(
                vad_input, (0, pad_size), mode='constant', value=0
            )
            vad_label = torch.nn.functional.pad(
                vad_label, (0, pad_size), mode='constant', value=0
            )
        
        vad_inputs.append(vad_input)
        vad_labels.append(vad_label)
    
    vad_inputs = torch.stack(vad_inputs)
    vad_labels = torch.stack(vad_labels)
    
    return {
        'utt_ids': utt_ids,
        # AEC任务数据
        'aec': {
            'input': aec_inputs,
            'target_mag': aec_target_mags,
            'target_phase': aec_target_phases,
            'mic_phase': mic_phases,
        },
        # VAD任务数据
        'vad': {
            'input': vad_inputs,
            'target_labels': vad_labels,
        }
    }


def create_scp_dataloader(microphone_scp: str,
                          far_end_scp: str,
                          near_end_scp: str,
                          vad_labels_scp: str,
                          batch_size: int = 16,
                          shuffle: bool = True,
                          num_workers: int = 4,
                          n_fft: int = 512,
                          hop_length: int = 128) -> DataLoader:
    """
    创建基于SCP文件的数据加载器
    
    Args:
        microphone_scp: 麦克风信号SCP文件路径
        far_end_scp: 远端信号SCP文件路径
        near_end_scp: 近端纯净语音SCP文件路径
        vad_labels_scp: VAD标签SCP文件路径
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        n_fft: STFT的FFT大小
        hop_length: STFT的帧移
    
    Returns:
        DataLoader实例
    """
    dataset = SCPDataset(
        microphone_scp=microphone_scp,
        far_end_scp=far_end_scp,
        near_end_scp=near_end_scp,
        vad_labels_scp=vad_labels_scp,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_scp_batch,
        pin_memory=True,
    )
    
    return dataloader
