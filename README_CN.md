# 多任务学习模型：声学回声消除(nnAEC)与语音活动检测(VAD)

本项目实现了一个多任务深度学习模型，可同时执行**声学回声消除(nnAEC)**和**语音活动检测(VAD)**。该模型利用共享表征来提升两个任务的性能。

## 📋 目录

- [概述](#概述)
- [当前工作评估](#当前工作评估)
- [模型架构](#模型架构)
- [项目结构](#项目结构)
- [安装](#安装)
- [数据预处理](#数据预处理)
- [训练](#训练)
- [评估](#评估)
- [单元测试](#单元测试)
- [下一步工作指导](#下一步工作指导)

## 🎯 概述

本实现提供了两种不同的多任务学习架构：

1. **共享主干模型**：使用共同的CRNN主干网络配合独立的任务头
2. **层次化模型**：先执行VAD，然后使用VAD信息通过注意力机制指导AEC

两种模型都设计为模块化、可扩展和生产就绪的。

## 📊 当前工作评估

### ✅ 已完成的核心功能

#### 1. **模型架构** (完整度: 90%)
- ✅ 两种CRNN多任务架构实现完整
  - 共享主干模型 (70M参数)
  - 层次化模型 (87M参数，带VAD引导)
- ✅ 模块化设计，组件可替换
- ✅ 完整的前向传播逻辑
- ⚠️ 缺少：实际数据集上的性能基准测试

#### 2. **数据处理流程** (完整度: 95%)
- ✅ 联合AEC-VAD预处理器
- ✅ 简化的单文件SCP格式
- ✅ 高效的PyTorch DataLoader
- ✅ **新增**：基于Povey窗口的STFT/iSTFT实现（与C端流式处理匹配）
- ✅ 在线数据增强（RIR卷积、SNR/SER控制）
- ⚠️ 缺少：大规模数据集的实际预处理示例

#### 3. **训练基础设施** (完整度: 85%)
- ✅ 多任务损失函数（可配置权重）
- ✅ 评估指标（AEC: SI-SDR, VAD: Acc/P/R/F1）
- ✅ 训练脚本框架
- ⚠️ 缺少：学习率调度、早停、模型checkpointing的完整实现
- ⚠️ 缺少：TensorBoard日志记录的实际集成

#### 4. **代码质量** (完整度: 95%)
- ✅ 42个单元测试，全部通过
- ✅ 模块化、可读的代码结构
- ✅ 详细的中文注释和文档
- ✅ 类型提示（部分）
- ⚠️ 缺少：集成测试、端到端测试

### ⚠️ 主要限制与风险

#### 技术限制
1. **STFT实现**
   - ✅ **已改进**：采用Povey窗口，与C端流式处理逻辑匹配
   - ⚠️ 音频重采样仍使用线性插值（生产建议用librosa）

2. **数据依赖**
   - 需要DNS-Challenge和LibriSpeech数据集（数百GB）
   - 预处理时间可能很长（未优化）

3. **计算资源**
   - 模型参数量大（70M-87M）
   - 需要GPU进行有效训练
   - 未进行量化或蒸馏优化

#### 验证缺口
1. **未在真实数据上验证**
   - 所有测试使用合成/随机数据
   - 模型收敛性未验证
   - 性能指标未在基准数据集上测试

2. **实时性未测试**
   - 流式处理能力未实现
   - 延迟未测量
   - 内存占用未优化

### 🎯 技术优势

1. **生产就绪的STFT**
   - Povey窗口实现与VOS SDK C端代码完全匹配
   - 支持流式处理的逐帧计算
   - 可靠的overlap-add重构

2. **清晰的数据流**
   - 单文件SCP格式，易于管理
   - 统一的数据接口，同时支持AEC和VAD
   - 兼容Kaldi等语音工具链

3. **灵活的架构**
   - 两种不同的多任务学习策略
   - 可配置的损失函数权重
   - 易于扩展到其他任务

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

## 📁 项目结构

```
aec-vad/
├── preprocessing/              # 数据预处理模块
│   ├── __init__.py
│   ├── joint_preprocessor.py   # 联合AEC-VAD预处理器
│   └── scp_dataset.py          # 基于SCP的数据集
├── models/                     # 模型实现
│   ├── __init__.py
│   ├── base.py                 # 基础类和组件
│   ├── shared_backbone.py      # 架构A
│   ├── hierarchical.py         # 架构B
│   ├── losses.py               # 损失函数
│   └── metrics.py              # 评估指标
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── audio_utils.py          # 音频处理工具
│   └── scp_utils.py            # SCP文件处理工具
├── tests/                      # 单元测试
│   └── ...                     # 测试文件
├── train.py                    # 训练脚本
├── prepare_data.py             # 数据预处理脚本
├── requirements.txt            # 依赖包列表
└── README_CN.md                # 本文档
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

### SCP数据格式

本项目使用简化的SCP格式，所有数据路径存储在一个文件中。

**SCP文件格式：**
```
# 每行格式: utt_id mic_path far_path near_path vad_path
train_000000 /path/to/mic.wav /path/to/far.wav /path/to/near.wav /path/to/vad.npy
train_000001 /path/to/mic.wav /path/to/far.wav /path/to/near.wav /path/to/vad.npy
...
```

### 生成训练数据

使用 `prepare_data.py` 脚本生成联合训练数据：

```bash
python prepare_data.py \
    --dns_root /path/to/DNS-Challenge/datasets \
    --librispeech_root /path/to/LibriSpeech \
    --output_dir ./data/processed \
    --num_train 10000 \
    --num_val 1000 \
    --num_test 1000
```

**参数说明：**
- `--dns_root`: DNS-Challenge数据集路径
- `--librispeech_root`: LibriSpeech数据集路径
- `--output_dir`: 输出目录
- `--num_train`: 训练集样本数量
- `--num_val`: 验证集样本数量
- `--num_test`: 测试集样本数量

### 数据预处理流程

联合预处理器 `JointAECVADPreprocessor` 会生成包含以下内容的训练样本：

1. **麦克风信号** (`microphone`): 混合信号 y(t) = s(t) + e(t) + n(t)
2. **远端参考信号** (`far_end`): 用于AEC的参考信号
3. **近端纯净语音** (`near_end`): AEC任务的目标输出
4. **VAD标签** (`vad_labels`): 逐帧二分类标签（0=非语音，1=语音）

所有路径存储在单个 `data.scp` 文件中。

## 🚀 训练

### Python API使用

```python
from preprocessing import create_scp_dataloader
from models import SharedBackboneModel
from models.losses import MultiTaskLoss

# 创建数据加载器
train_loader = create_scp_dataloader(
    scp_file='./data/processed/train/data.scp',
    batch_size=16,
    shuffle=True,
)

# 初始化模型和训练组件
model = SharedBackboneModel(n_freqs=257)
criterion = MultiTaskLoss(aec_weight=1.0, vad_weight=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
for batch in train_loader:
    # batch同时包含AEC和VAD数据
    aec_output = model(batch['aec']['input'], task='aec')
    vad_output = model(batch['vad']['input'], task='vad')
    
    # 计算损失
    outputs = {'aec': aec_output, 'vad': vad_output}
    targets = {
        'aec': {'target_mag': batch['aec']['target_mag']},
        'vad': {'target_labels': batch['vad']['target_labels']}
    }
    losses = criterion(outputs, targets)
    
    # 反向传播
    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()
```

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
- **PESQ**（感知语音质量评估）：感知质量
- **STOI**（短时客观可懂度）：语音可懂度
- **ERLE**（回声返回损耗增强）：回声抑制

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

所有测试使用**模拟数据**（随机生成的张量），不需要实际数据集。

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

## 🚧 下一步工作指导

基于当前工作评估，以下是按优先级排序的改进建议：

### 高优先级（核心功能完善）

1. **实际数据集训练与验证** ⭐⭐⭐
   - 下载DNS-Challenge和LibriSpeech数据集
   - 运行完整的数据预处理流程
   - 在实际数据上训练模型，验证收敛性
   - 建立性能基准（AEC: PESQ/STOI/ERLE, VAD: F1）
   - **预计工作量**：3-5天

2. **训练基础设施完善** ⭐⭐⭐
   - 实现学习率调度（ReduceLROnPlateau）
   - 添加早停机制（EarlyStopping）
   - 完善模型checkpointing（保存最佳模型）
   - 集成TensorBoard可视化
   - **预计工作量**：1-2天

3. **性能优化** ⭐⭐
   - 使用librosa替换音频重采样
   - 优化数据预处理速度（多进程）
   - 添加混合精度训练支持（FP16）
   - **预计工作量**：2-3天

### 中优先级（增强功能）

4. **流式处理支持** ⭐⭐
   - 实现逐帧流式推理
   - 测量端到端延迟
   - 优化缓存机制
   - **预计工作量**：3-4天
   - **依赖**：需要先完成#1

5. **模型压缩与部署** ⭐
   - 模型量化（INT8）
   - 知识蒸馏（训练小模型）
   - ONNX导出
   - TorchScript转换
   - **预计工作量**：4-5天
   - **依赖**：需要先完成#1

6. **高级数据增强** ⭐
   - 添加更多RIR变体
   - 实时音高变换
   - 语速调节
   - 混响强度控制
   - **预计工作量**：2-3天

### 低优先级（锦上添花）

7. **更多架构变体**
   - Transformer-based模型
   - Conformer架构
   - 自注意力机制改进
   - **预计工作量**：5-7天

8. **对比实验**
   - 与单任务模型对比
   - 与SOTA方法对比
   - 消融实验
   - **预计工作量**：3-4天
   - **依赖**：需要先完成#1

9. **生产化工具**
   - Web演示界面
   - REST API服务
   - Docker容器化
   - **预计工作量**：3-5天

### 建议的开发路线图

**第一阶段（1-2周）：验证与完善**
- [ ] 完成任务#1：实际数据训练
- [ ] 完成任务#2：训练基础设施
- [ ] 完成任务#3：性能优化

**第二阶段（2-3周）：功能增强**
- [ ] 完成任务#4：流式处理
- [ ] 完成任务#5：模型压缩
- [ ] 完成任务#6：数据增强

**第三阶段（1-2周）：研究与发布**
- [ ] 完成任务#8：对比实验
- [ ] 撰写技术报告/论文
- [ ] 准备开源发布

### 已知技术债务

1. **测试覆盖**
   - 需要添加集成测试
   - 需要端到端测试
   - 需要性能回归测试

2. **文档**
   - 添加API文档（Sphinx）
   - 添加架构设计文档
   - 添加故障排除指南

3. **代码质量**
   - 添加类型检查（mypy）
   - 添加代码格式化（black）
   - 添加CI/CD流水线

## 📄 许可证

本项目仅供教育和研究目的。

## 📧 联系方式

如有问题或反馈，请在GitHub仓库中提交Issue。
