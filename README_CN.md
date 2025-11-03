# 多任务学习模型：声学回声消除(nnAEC)与语音活动检测(VAD)

本项目实现了一个多任务深度学习模型，可同时执行**声学回声消除(nnAEC)**和**语音活动检测(VAD)**。该模型利用共享表征来提升两个任务的性能。

## 📋 目录

- [概述](#概述)
- [模型架构](#模型架构)
- [项目结构](#项目结构)
- [安装](#安装)
- [数据预处理](#数据预处理)
- [训练](#训练)
- [评估](#评估)
- [单元测试](#单元测试)
- [实验结果](#实验结果)

## 🎯 概述

本实现探索了两种不同的多任务学习架构：

1. **共享主干模型**：使用共同的CRNN主干网络配合独立的任务头
2. **层次化模型**：先执行VAD，然后使用VAD信息通过注意力机制指导AEC

两种模型都设计为模块化、可扩展和生产就绪的。

## 🏗️ 模型架构

### 架构A：共享主干模型

```
输入(AEC/VAD) 
    ↓
特征提取器（任务特定）
    ↓
共享CRNN主干
    ├─→ AEC头 → 增强语音幅度谱
    └─→ VAD头 → 语音活动概率
```

**主要特点：**
- 共享CRNN主干用于通用特征提取
- 针对不同输入通道的独立特征提取器
- 独立的任务特定头
- 参数共享促进学习通用表征

**架构细节：**
- 特征提取器：3层CNN（64通道）
- 主干网络：2层CNN（128通道）+ 2层双向LSTM（256隐藏单元）
- AEC头：2层MLP → 幅度谱输出
- VAD头：3层MLP → 逐帧语音概率

### 架构B：层次化/渐进式模型

```
输入(AEC) → 特征提取器 → AEC主干 → 注意力门 → AEC头
                                        ↑
输入(VAD) → 特征提取器 → VAD主干 → VAD头
                                        ↓
                                    VAD引导
```

**主要特点：**
- VAD分支提前处理特征
- VAD预测通过注意力门控指导AEC
- 渐进式细化：粗粒度VAD → 细粒度AEC
- 注意力机制使AEC聚焦于语音区域

**架构细节：**
- AEC和VAD使用独立的主干网络（不同容量）
- 注意力门使用VAD概率调制AEC特征
- VAD主干：更轻量（64卷积通道，128 LSTM隐藏单元）
- AEC主干：更重（128卷积通道，256 LSTM隐藏单元）

## 📁 项目结构

```
aec-vad/
├── preprocessing/              # 数据预处理模块
│   ├── __init__.py
│   ├── aec_preprocessor.py      # AEC数据生成
│   ├── vad_preprocessor.py      # VAD数据生成
│   ├── dataloader.py            # 多任务数据加载器
│   ├── joint_preprocessor.py    # 联合AEC-VAD预处理器（新增）
│   └── scp_dataset.py           # 基于SCP的数据集（新增）
├── models/                      # 模型实现
│   ├── __init__.py
│   ├── base.py                  # 基础类和组件
│   ├── shared_backbone.py       # 架构A
│   ├── hierarchical.py          # 架构B
│   ├── losses.py                # 损失函数
│   └── metrics.py               # 评估指标
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── audio_utils.py           # 音频处理工具
│   └── scp_utils.py             # SCP文件处理工具（新增）
├── tests/                       # 单元测试
│   ├── __init__.py
│   ├── test_audio_utils.py      # 音频工具测试
│   ├── test_models.py           # 模型架构测试
│   ├── test_losses.py           # 损失函数测试
│   ├── test_metrics.py          # 指标测试
│   └── test_scp_dataset.py      # SCP数据集测试（新增）
├── train.py                     # 训练脚本
├── prepare_data.py              # 数据预处理脚本（新增）
├── requirements.txt             # 依赖包列表
├── README.md                    # 英文文档
├── README_CN.md                 # 本文档（中文）
└── SCP_README_CN.md            # SCP数据处理指南（新增）
```

## 🔧 安装

### 环境要求

- Python 3.8或更高版本
- PyTorch 2.0或更高版本
- CUDA（可选，用于GPU训练）

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/KaixingDing/aec-vad.git
cd aec-vad
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 📊 数据预处理

### 方法1：使用联合预处理器（推荐）

新的联合预处理器可以生成同时用于AEC和VAD任务的数据，并保存为SCP格式。

```bash
python prepare_data.py \
    --dns_root /path/to/DNS-Challenge/datasets \
    --librispeech_root /path/to/LibriSpeech \
    --output_dir ./data/processed \
    --num_train 10000 \
    --num_val 1000 \
    --num_test 1000
```

详细使用方法请参见 [SCP数据处理指南](SCP_README_CN.md)。

### 方法2：使用原始预处理器

#### AEC数据预处理

AEC预处理器从DNS-Challenge数据集生成训练样本：

**流程：**
1. 从 `clean/read_speech` 或 `clean/emotional_speech` 中选择近端语音（目标）
2. 从 `clean/` 中选择远端语音（与近端不同）
3. 从 `impulse_responses/` 中选择房间脉冲响应(RIR)
4. 通过卷积生成回声信号
5. 从 `noise/` 目录添加背景噪声
6. 合成麦克风信号：`y(t) = s(t) + e(t) + n(t)`

**使用示例：**
```python
from preprocessing import AECDataPreprocessor

preprocessor = AECDataPreprocessor(
    dns_root='/path/to/DNS-Challenge/datasets',
    sample_rate=16000,
    ser_range=(-5.0, 15.0),  # 信号回声比范围(dB)
    snr_range=(0.0, 30.0),   # 信噪比范围(dB)
    duration=4.0,            # 样本时长(秒)
)

preprocessor.scan_files()
mic, far_end, near_end = preprocessor.generate_sample()
```

#### VAD数据预处理

VAD预处理器从LibriSpeech和DNS-Challenge生成样本：

**流程：**
1. **正样本**：从LibriSpeech获取纯净语音
2. **负样本**：从DNS-Challenge获取噪声
3. **混合样本**：语音+噪声或拼接[噪声]+[语音]+[噪声]
4. 生成逐帧二分类标签(0=非语音，1=语音)

**使用示例：**
```python
from preprocessing import VADDataPreprocessor

preprocessor = VADDataPreprocessor(
    librispeech_root='/path/to/LibriSpeech',
    dns_root='/path/to/DNS-Challenge/datasets',
    sample_rate=16000,
    duration=4.0,
)

preprocessor.scan_files()
signal, labels = preprocessor.generate_sample()
```

## 🚀 训练

### 基础训练

使用SCP数据集训练共享主干模型：

```bash
python train.py \
    --model_type shared_backbone \
    --train_microphone_scp ./data/processed/train/microphone.scp \
    --train_far_end_scp ./data/processed/train/far_end.scp \
    --train_near_end_scp ./data/processed/train/near_end.scp \
    --train_vad_labels_scp ./data/processed/train/vad_labels.scp \
    --batch_size 16 \
    --num_epochs 100 \
    --output_dir ./outputs/shared_backbone
```

训练层次化模型：

```bash
python train.py \
    --model_type hierarchical \
    --train_microphone_scp ./data/processed/train/microphone.scp \
    --train_far_end_scp ./data/processed/train/far_end.scp \
    --train_near_end_scp ./data/processed/train/near_end.scp \
    --train_vad_labels_scp ./data/processed/train/vad_labels.scp \
    --batch_size 16 \
    --num_epochs 100 \
    --output_dir ./outputs/hierarchical
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_type` | shared_backbone | 模型架构(shared_backbone或hierarchical) |
| `--batch_size` | 16 | 批次大小 |
| `--num_epochs` | 100 | 训练轮数 |
| `--learning_rate` | 1e-3 | 学习率 |
| `--aec_weight` | 1.0 | AEC损失权重 |
| `--vad_weight` | 1.0 | VAD损失权重 |
| `--output_dir` | ./outputs | 输出目录 |

### 多任务损失函数

总损失是加权组合：

```
L_total = λ_AEC × L_AEC + λ_VAD × L_VAD

其中:
  L_AEC = w_mag × L_magnitude + w_sisdr × L_SI-SDR
  L_VAD = Binary Cross-Entropy
```

## 📈 评估

### AEC指标

- **SI-SDR**（尺度不变信号失真比）：衡量增强质量
- **PESQ**（感知语音质量评估）：感知质量（需要时域信号）
- **STOI**（短时客观可懂度）：语音可懂度（需要时域信号）
- **ERLE**（回声返回损耗增强）：回声抑制（需要时域信号）

*注意：完整的PESQ、STOI和ERLE需要时域重建和外部库（pesq、pystoi）。*

### VAD指标

- **准确率**：总体分类准确性
- **精确率**：预测为语音中实际为语音的比例
- **召回率**：实际语音被检测出的比例
- **F1分数**：精确率和召回率的调和平均

## 🧪 单元测试

运行所有测试：

```bash
pytest tests/ -v
```

运行特定测试模块：

```bash
# 测试音频工具
pytest tests/test_audio_utils.py -v

# 测试模型架构
pytest tests/test_models.py -v

# 测试SCP数据集
pytest tests/test_scp_dataset.py -v
```

**测试覆盖：**
- ✅ 音频处理工具（STFT、混音、RIR应用）
- ✅ 模型架构（前向传播、梯度流、设备转移）
- ✅ 损失函数（SI-SDR、幅度、多任务）
- ✅ 评估指标（AEC SI-SDR、VAD准确率/F1）
- ✅ SCP数据集加载和批处理

所有测试使用**模拟数据**（随机生成的张量），不需要实际数据集。

## 📊 实验结果

### 模型对比

| 模型 | 参数量 | AEC SI-SDR | VAD F1 | 训练时间 |
|------|--------|------------|--------|----------|
| 共享主干 | ~70M | 待测 | 待测 | 待测 |
| 层次化 | ~87M | 待测 | 待测 | 待测 |

*结果将在实际数据集训练后更新。*

### 架构分析

**共享主干模型：**
- ✅ 架构更简单
- ✅ 高效的参数共享
- ✅ 适合均衡的多任务学习
- ⚠️ 如果任务目标冲突可能表现较差

**层次化模型：**
- ✅ VAD引导改善了语音区域的AEC效果
- ✅ 更好地处理静音/纯噪声片段
- ✅ 更具可解释性（VAD → AEC流水线）
- ⚠️ 计算成本更高
- ⚠️ 串行处理可能增加延迟

## 🔬 研究见解

### 多任务学习的优势

1. **共享表征**：两个任务都受益于学习通用的声学特征
2. **数据效率**：共享主干比两个独立模型减少了总参数量
3. **正则化**：多任务学习起到隐式正则化作用
4. **任务协同**：VAD帮助AEC聚焦于语音区域，AEC帮助VAD应对噪声条件

### 未来改进

1. **模型架构**：
   - 探索基于Transformer的架构
   - 添加残差连接和跳跃连接
   - 研究U-Net风格的编码器-解码器

2. **损失函数**：
   - 添加感知损失（PESQ、STOI）
   - 探索对抗性损失以获得更自然的语音
   - 动态任务权重策略

3. **训练**：
   - 课程学习（从简单样本开始）
   - 数据增强（音高变换、时间拉伸）
   - 在线困难样本挖掘
   - 使用不同随机种子的独立验证数据加载器

4. **扩展**：
   - 添加更多任务（去噪、源分离）
   - 实时推理优化
   - 模型压缩和量化

5. **代码质量**：
   - 使用librosa.resample()替换线性插值重采样
   - 使用scipy.signal.windows.hann()替代已弃用的np.hanning()
   - 在音频重采样中实现适当的抗混叠

## ⚠️ 已知限制

- 音频重采样使用线性插值（为简单起见）；生产代码应使用适当的抗混叠重采样
- np.hanning()在较新的NumPy版本中已弃用；考虑使用scipy.signal替代方案
- 演示模式下训练和验证使用相同的数据加载器；实际训练建议使用独立的验证集
- STFT/iSTFT实现较基础；生产环境考虑使用torchaudio

## 📝 引用

如果您在研究中使用此代码，请引用：

```bibtex
@software{aec_vad_multitask,
  title={多任务学习模型：声学回声消除与语音活动检测},
  author={丁凯星},
  year={2025},
  url={https://github.com/KaixingDing/aec-vad}
}
```

## 📄 许可证

本项目仅供教育和研究目的。

## 🙏 致谢

- DNS-Challenge数据集提供的多样化声学场景
- LibriSpeech语料库提供的纯净语音数据
- PyTorch团队提供的优秀深度学习框架

## 📧 联系方式

如有问题或反馈，请在GitHub仓库中提交Issue。
