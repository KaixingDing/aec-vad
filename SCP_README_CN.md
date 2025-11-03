# 基于SCP文件的AEC-VAD联合训练数据处理指南

## 📖 概述

本文档介绍如何使用SCP（Script）文件格式来组织和加载AEC-VAD联合训练数据。SCP文件是语音处理领域（特别是Kaldi工具包）中广泛使用的数据索引格式。

## 🎯 设计目标

新的数据处理流程实现了以下目标：

1. **统一数据格式**：将AEC和VAD任务的数据整合到同一格式
2. **高效索引**：使用SCP文件快速定位和加载数据
3. **灵活性**：支持大规模数据集的分布式处理
4. **兼容性**：与现有语音处理工具链兼容

## 📦 数据组织结构

### SCP文件格式

每个SCP文件包含样本ID到文件路径的映射：

```
utt001 /path/to/data/utt001.wav
utt002 /path/to/data/utt002.wav
utt003 /path/to/data/utt003.wav
```

### 联合数据集结构

预处理后的数据集包含以下文件：

```
data/processed/
├── train/
│   ├── microphone/          # 麦克风信号目录
│   │   ├── train_000000.wav
│   │   ├── train_000001.wav
│   │   └── ...
│   ├── far_end/             # 远端参考信号目录
│   │   ├── train_000000.wav
│   │   └── ...
│   ├── near_end/            # 近端纯净语音目录（AEC目标）
│   │   ├── train_000000.wav
│   │   └── ...
│   ├── vad_labels/          # VAD标签目录
│   │   ├── train_000000.npy
│   │   └── ...
│   ├── microphone.scp       # 麦克风信号索引
│   ├── far_end.scp          # 远端信号索引
│   ├── near_end.scp         # 近端语音索引
│   └── vad_labels.scp       # VAD标签索引
├── val/                     # 验证集（结构同上）
└── test/                    # 测试集（结构同上）
```

## 🚀 使用方法

### 1. 数据预处理

使用 `prepare_data.py` 脚本生成联合训练数据：

```bash
python prepare_data.py \
    --dns_root /path/to/DNS-Challenge/datasets \
    --librispeech_root /path/to/LibriSpeech \
    --output_dir ./data/processed \
    --num_train 10000 \
    --num_val 1000 \
    --num_test 1000 \
    --sample_rate 16000 \
    --duration 4.0
```

**参数说明：**
- `--dns_root`: DNS-Challenge数据集路径
- `--librispeech_root`: LibriSpeech数据集路径
- `--output_dir`: 输出目录
- `--num_train`: 训练集样本数量
- `--num_val`: 验证集样本数量
- `--num_test`: 测试集样本数量
- `--sample_rate`: 采样率（Hz）
- `--duration`: 每个样本的时长（秒）

### 2. 创建数据加载器

```python
from preprocessing import create_scp_dataloader

# 创建训练数据加载器
train_loader = create_scp_dataloader(
    microphone_scp='./data/processed/train/microphone.scp',
    far_end_scp='./data/processed/train/far_end.scp',
    near_end_scp='./data/processed/train/near_end.scp',
    vad_labels_scp='./data/processed/train/vad_labels.scp',
    batch_size=16,
    shuffle=True,
    num_workers=4,
)

# 创建验证数据加载器
val_loader = create_scp_dataloader(
    microphone_scp='./data/processed/val/microphone.scp',
    far_end_scp='./data/processed/val/far_end.scp',
    near_end_scp='./data/processed/val/near_end.scp',
    vad_labels_scp='./data/processed/val/vad_labels.scp',
    batch_size=16,
    shuffle=False,
    num_workers=4,
)
```

### 3. 模型训练

```python
from models import SharedBackboneModel
from models.losses import MultiTaskLoss
from models.metrics import MultiTaskMetrics

# 初始化模型
model = SharedBackboneModel(n_freqs=257)

# 初始化损失函数和评估指标
criterion = MultiTaskLoss(aec_weight=1.0, vad_weight=1.0)
metrics = MultiTaskMetrics()

# 训练循环
for epoch in range(num_epochs):
    for batch in train_loader:
        # 批次已包含AEC和VAD所有必要数据
        # batch['aec']['input']: AEC输入（麦克风+远端）
        # batch['aec']['target_mag']: AEC目标（近端语音幅度）
        # batch['vad']['input']: VAD输入（麦克风）
        # batch['vad']['target_labels']: VAD目标标签
        
        # AEC任务前向传播
        aec_output = model(batch['aec']['input'], task='aec')
        
        # VAD任务前向传播
        vad_output = model(batch['vad']['input'], task='vad')
        
        # 计算损失
        outputs = {'aec': aec_output, 'vad': vad_output}
        targets = {
            'aec': {'target_mag': batch['aec']['target_mag']},
            'vad': {'target_labels': batch['vad']['target_labels']}
        }
        losses = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
```

## 📊 数据批次结构

从SCP数据加载器获取的每个批次包含以下结构：

```python
batch = {
    'utt_ids': ['train_000000', 'train_000001', ...],  # 样本ID列表
    
    'aec': {
        'input': torch.Tensor,         # (batch, 2, n_freqs, n_frames)
                                       # 2通道：麦克风 + 远端参考
        'target_mag': torch.Tensor,    # (batch, n_freqs, n_frames)
                                       # AEC目标：近端纯净语音幅度谱
        'target_phase': torch.Tensor,  # (batch, n_freqs, n_frames)
                                       # 近端语音相位谱
        'mic_phase': torch.Tensor,     # (batch, n_freqs, n_frames)
                                       # 麦克风信号相位谱
    },
    
    'vad': {
        'input': torch.Tensor,         # (batch, 1, n_freqs, n_frames)
                                       # 1通道：麦克风幅度谱
        'target_labels': torch.Tensor, # (batch, n_frames)
                                       # VAD目标：逐帧二分类标签
    }
}
```

## 🔧 核心API

### JointAECVADPreprocessor

联合预处理器类，用于生成包含AEC和VAD所有信息的训练样本。

```python
from preprocessing import JointAECVADPreprocessor

preprocessor = JointAECVADPreprocessor(
    dns_root='/path/to/DNS-Challenge',
    librispeech_root='/path/to/LibriSpeech',
    sample_rate=16000,
    ser_range=(-5.0, 15.0),   # 信号回声比范围
    snr_range=(0.0, 30.0),     # 信噪比范围
    duration=4.0,              # 样本时长
)

# 扫描数据集
preprocessor.scan_files()

# 生成单个样本
sample = preprocessor.generate_joint_sample()
# sample包含: microphone, far_end, near_end, vad_labels

# 批量生成并保存为SCP格式
preprocessor.preprocess_dataset(
    num_samples=10000,
    output_dir='./data/processed',
    split='train'
)
```

### SCPDataset

基于SCP文件的PyTorch数据集类。

```python
from preprocessing import SCPDataset

dataset = SCPDataset(
    microphone_scp='./data/processed/train/microphone.scp',
    far_end_scp='./data/processed/train/far_end.scp',
    near_end_scp='./data/processed/train/near_end.scp',
    vad_labels_scp='./data/processed/train/vad_labels.scp',
    n_fft=512,
    hop_length=128,
)

# 获取样本
sample = dataset[0]  # 返回包含AEC和VAD数据的字典
```

### SCP工具函数

```python
from utils.scp_utils import read_scp, write_scp, create_scp_list

# 读取SCP文件
scp_dict = read_scp('data.scp')
# 返回: {'utt001': '/path/to/utt001.wav', ...}

# 写入SCP文件
write_scp(scp_dict, 'output.scp')

# 从目录创建SCP文件
scp_dict = create_scp_list(
    data_dir='/path/to/wavs',
    pattern='*.wav',
    output_scp='output.scp'
)
```

## ✨ 优势

### 1. 统一数据接口

- 单个数据加载器同时提供AEC和VAD所需的所有数据
- 无需在训练循环中进行复杂的数据整合
- 自动处理STFT特征提取

### 2. 高效数据管理

- SCP文件提供O(1)的样本索引
- 支持大规模数据集（数百万样本）
- 便于数据集划分和管理

### 3. 灵活性

- 可以轻松添加新的数据源
- 支持自定义数据增强
- 兼容Kaldi等语音工具

### 4. 可复现性

- 固定的数据集划分（通过SCP文件）
- 便于实验对比和结果复现

## 🎓 示例：完整训练流程

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""完整的SCP数据集训练示例"""

import torch
from preprocessing import create_scp_dataloader
from models import SharedBackboneModel
from models.losses import MultiTaskLoss
from models.metrics import MultiTaskMetrics

# 1. 创建数据加载器
train_loader = create_scp_dataloader(
    microphone_scp='./data/processed/train/microphone.scp',
    far_end_scp='./data/processed/train/far_end.scp',
    near_end_scp='./data/processed/train/near_end.scp',
    vad_labels_scp='./data/processed/train/vad_labels.scp',
    batch_size=16,
    shuffle=True,
)

val_loader = create_scp_dataloader(
    microphone_scp='./data/processed/val/microphone.scp',
    far_end_scp='./data/processed/val/far_end.scp',
    near_end_scp='./data/processed/val/near_end.scp',
    vad_labels_scp='./data/processed/val/vad_labels.scp',
    batch_size=16,
    shuffle=False,
)

# 2. 初始化模型和训练组件
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SharedBackboneModel(n_freqs=257).to(device)
criterion = MultiTaskLoss(aec_weight=1.0, vad_weight=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
metrics = MultiTaskMetrics()

# 3. 训练循环
num_epochs = 100

for epoch in range(num_epochs):
    # 训练
    model.train()
    for batch in train_loader:
        # 移动数据到设备
        aec_input = batch['aec']['input'].to(device)
        aec_target = batch['aec']['target_mag'].to(device)
        vad_input = batch['vad']['input'].to(device)
        vad_target = batch['vad']['target_labels'].to(device)
        
        # 前向传播
        aec_output = model(aec_input, task='aec')
        vad_output = model(vad_input, task='vad')
        
        # 计算损失
        outputs = {'aec': aec_output, 'vad': vad_output}
        targets = {
            'aec': {'target_mag': aec_target},
            'vad': {'target_labels': vad_target}
        }
        losses = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
    
    # 验证
    model.eval()
    metrics.reset()
    
    with torch.no_grad():
        for batch in val_loader:
            aec_input = batch['aec']['input'].to(device)
            aec_target = batch['aec']['target_mag'].to(device)
            vad_input = batch['vad']['input'].to(device)
            vad_target = batch['vad']['target_labels'].to(device)
            
            aec_output = model(aec_input, task='aec')
            vad_output = model(vad_input, task='vad')
            
            outputs = {'aec': aec_output, 'vad': vad_output}
            targets = {
                'aec': {'target_mag': aec_target},
                'vad': {'target_labels': vad_target}
            }
            
            metrics.update(outputs, targets)
    
    # 打印评估结果
    val_metrics = metrics.compute()
    print(f"Epoch {epoch + 1}:")
    print(f"  AEC SI-SDR: {val_metrics['aec']['si_sdr']:.2f} dB")
    print(f"  VAD F1: {val_metrics['vad']['f1_score']:.4f}")
```

## 📝 注意事项

1. **内存管理**：大规模数据集建议使用多个worker进程（`num_workers > 0`）
2. **数据平衡**：确保AEC和VAD样本分布均衡
3. **路径一致性**：SCP文件中的路径必须有效
4. **采样率匹配**：所有音频文件必须使用相同的采样率

## 🔗 相关文档

- [主README](README_CN.md)：项目总体介绍
- [模型架构文档](models/README_CN.md)：模型设计详解
- [训练指南](docs/training_cn.md)：完整训练流程

## 📞 技术支持

如有问题，请在GitHub仓库中提交Issue。
