"""基于SCP文件的联合AEC-VAD数据集测试"""

import pytest
import torch
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path

from preprocessing.scp_dataset import SCPDataset, collate_scp_batch, create_scp_dataloader
from utils.scp_utils import write_scp


def create_mock_scp_data(tmp_dir):
    """创建模拟SCP数据"""
    tmp_path = Path(tmp_dir)
    
    # 创建子目录
    mic_dir = tmp_path / 'microphone'
    far_dir = tmp_path / 'far_end'
    near_dir = tmp_path / 'near_end'
    vad_dir = tmp_path / 'vad_labels'
    
    for d in [mic_dir, far_dir, near_dir, vad_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 生成模拟数据
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    
    mic_scp = {}
    far_scp = {}
    near_scp = {}
    vad_scp = {}
    
    for i in range(5):
        utt_id = f'test_{i:03d}'
        
        # 生成随机音频
        mic_audio = np.random.randn(num_samples) * 0.1
        far_audio = np.random.randn(num_samples) * 0.1
        near_audio = np.random.randn(num_samples) * 0.1
        
        # 生成VAD标签
        vad_labels = np.random.randint(0, 2, 100)
        
        # 保存文件
        mic_path = mic_dir / f'{utt_id}.wav'
        far_path = far_dir / f'{utt_id}.wav'
        near_path = near_dir / f'{utt_id}.wav'
        vad_path = vad_dir / f'{utt_id}.npy'
        
        sf.write(mic_path, mic_audio, sample_rate)
        sf.write(far_path, far_audio, sample_rate)
        sf.write(near_path, near_audio, sample_rate)
        np.save(vad_path, vad_labels)
        
        # 添加到SCP字典
        mic_scp[utt_id] = str(mic_path.absolute())
        far_scp[utt_id] = str(far_path.absolute())
        near_scp[utt_id] = str(near_path.absolute())
        vad_scp[utt_id] = str(vad_path.absolute())
    
    # 写入SCP文件
    mic_scp_path = tmp_path / 'microphone.scp'
    far_scp_path = tmp_path / 'far_end.scp'
    near_scp_path = tmp_path / 'near_end.scp'
    vad_scp_path = tmp_path / 'vad_labels.scp'
    
    write_scp(mic_scp, mic_scp_path)
    write_scp(far_scp, far_scp_path)
    write_scp(near_scp, near_scp_path)
    write_scp(vad_scp, vad_scp_path)
    
    return mic_scp_path, far_scp_path, near_scp_path, vad_scp_path


def test_scp_dataset_creation():
    """测试SCP数据集创建"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 创建模拟数据
        mic_scp, far_scp, near_scp, vad_scp = create_mock_scp_data(tmp_dir)
        
        # 创建数据集
        dataset = SCPDataset(
            microphone_scp=str(mic_scp),
            far_end_scp=str(far_scp),
            near_end_scp=str(near_scp),
            vad_labels_scp=str(vad_scp),
            n_fft=512,
            hop_length=128,
        )
        
        # 检查数据集大小
        assert len(dataset) == 5
        
        # 获取一个样本
        sample = dataset[0]
        
        # 检查样本包含所有必要的键
        assert 'utt_id' in sample
        assert 'aec_input' in sample
        assert 'aec_target_mag' in sample
        assert 'vad_input' in sample
        assert 'vad_labels' in sample
        
        # 检查张量形状
        assert sample['aec_input'].shape[0] == 2  # 2通道（麦克风+远端）
        assert sample['vad_input'].shape[0] == 1  # 1通道（仅麦克风）


def test_scp_dataset_batch():
    """测试SCP数据集批处理"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 创建模拟数据
        mic_scp, far_scp, near_scp, vad_scp = create_mock_scp_data(tmp_dir)
        
        # 创建数据集
        dataset = SCPDataset(
            microphone_scp=str(mic_scp),
            far_end_scp=str(far_scp),
            near_end_scp=str(near_scp),
            vad_labels_scp=str(vad_scp),
        )
        
        # 获取多个样本
        batch = [dataset[i] for i in range(3)]
        
        # 整理批次
        collated = collate_scp_batch(batch)
        
        # 检查批次结构
        assert 'utt_ids' in collated
        assert 'aec' in collated
        assert 'vad' in collated
        
        # 检查AEC数据
        assert 'input' in collated['aec']
        assert 'target_mag' in collated['aec']
        assert collated['aec']['input'].shape[0] == 3  # 批次大小
        
        # 检查VAD数据
        assert 'input' in collated['vad']
        assert 'target_labels' in collated['vad']
        assert collated['vad']['input'].shape[0] == 3  # 批次大小


def test_scp_dataloader():
    """测试SCP数据加载器"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 创建模拟数据
        mic_scp, far_scp, near_scp, vad_scp = create_mock_scp_data(tmp_dir)
        
        # 创建数据加载器
        dataloader = create_scp_dataloader(
            microphone_scp=str(mic_scp),
            far_end_scp=str(far_scp),
            near_end_scp=str(near_scp),
            vad_labels_scp=str(vad_scp),
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        
        # 获取一个批次
        batch = next(iter(dataloader))
        
        # 检查批次包含AEC和VAD数据
        assert 'aec' in batch
        assert 'vad' in batch
        
        # 检查批次大小
        assert batch['aec']['input'].shape[0] == 2
        assert batch['vad']['input'].shape[0] == 2


def test_scp_dataset_with_model():
    """测试SCP数据集与模型配合使用"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 创建模拟数据
        mic_scp, far_scp, near_scp, vad_scp = create_mock_scp_data(tmp_dir)
        
        # 创建数据加载器
        dataloader = create_scp_dataloader(
            microphone_scp=str(mic_scp),
            far_end_scp=str(far_scp),
            near_end_scp=str(near_scp),
            vad_labels_scp=str(vad_scp),
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        
        # 模拟模型前向传播
        from models import SharedBackboneModel
        
        model = SharedBackboneModel(n_freqs=257)
        
        for batch in dataloader:
            # AEC任务
            aec_output = model(batch['aec']['input'], task='aec')
            assert 'magnitude' in aec_output
            
            # VAD任务
            vad_output = model(batch['vad']['input'], task='vad')
            assert 'logits' in vad_output
            
            break  # 只测试一个批次


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
