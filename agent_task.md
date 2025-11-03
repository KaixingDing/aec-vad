## 📅 项目任务书：基于nnAEC与VAD的多任务学习模型研究

### 1\. 🎯 项目目标

本项目旨在研究、设计并实现一个**多任务深度学习模型**，该模型需能够**同时**执行**声学回声消除 (nnAEC)** 和**语音活动检测 (VAD)**。

核心目标是利用单一模型实现两种功能，探索共享表征（Shared Representation）对两个任务的潜在增益，并最终提供一个高效、可扩展的实现方案。

### 2\. 📋 核心研究与开发任务

#### 任务1：文献调研与模型选型

1.  **调研 (Research):**
      * 调研当前SOTA（State-of-the-Art）的nnAEC模型架构（例如基于U-Net、CRNN、Transformer等的模型）。
      * 调研高效的VAD模型架构。
      * 调研将AEC和VAD结合的多任务学习（Multi-Task Learning, MTL）策略，特别是参数共享和损失函数设计。
2.  **选型 (Selection):**
      * 基于调研，请提出**至少两种**不同的多任务模型架构方案以供选择。
      * **方案示例（仅供启发，请自行研究）：**
          * **方案A (Shared Backbone):** 一个共享的编码器（如CRNN）提取时频域特征，然后接入两个独立（或部分连接）的解码器/任务头（Head），分别用于AEC（输出增强后的近端语音）和VAD（输出逐帧的语音活动概率）。
          * **方案B (Hierarchical/Progressive):** 模型首先进行粗粒度的特征提取和VAD判断，然后利用VAD的信息作为辅助（例如通过Attention或Gating机制），引导AEC任务更专注于处理包含语音的帧。

#### 任务2：数据预处理

你需要编写健壮的（Robust）数据预处理脚本，用于生成两个任务的训练样本。

1.  **AEC数据 (基于 DNS-Challenge):**

      * **目标：** 合成 (近端语音, 远端语音, 麦克风录音) 即 $(s(t), x(t), y(t))$ 的数据对。
      * **步骤：**
        1.  从 `clean/read_speech` 或 `clean/emotional_speech` 中随机选择一个文件作为**近端语音 $s(t)$**（target）。
        2.  从 `clean/` 中（选择不同于1的）随机选择另一个文件作为**远端语音 $x(t)$**。
        3.  从 `impulse_responses/` 中随机选择一个房间脉冲响应 (RIR) $h(t)$。
        4.  通过卷积 $x(t) * h(t)$ 生成**回声信号 $e(t)$**。
        5.  从 `noise/` 目录中随机选择一个**背景噪声 $n(t)$**。
        6.  合成**麦克风信号 $y(t)$**：$y(t) = s(t) + e(t) + n(t)$。（注意：需要控制 $s(t)$、 $e(t)$ 和 $n(t)$ 之间的信噪比(SNR)和信回比(SER)，使其符合真实场景分布）。
      * **输入：** $y(t)$ (麦克风信号) 和 $x(t)$ (远端参考信号)。
      * **输出：** $s(t)$ (近端语音信号)。

2.  **VAD数据 (基于 LibriSpeech 及 DNS-Challenge):**

      * **目标：** 生成 (混合音频, 逐帧VAD标签) 的数据对。
      * **步骤：**
        1.  **正样本 (Speech):** 使用 `LibriSpeech` 中的 `train-clean-100` 等目录下的音频。这些音频基本（或全部）是纯净语音。
        2.  **负样本 (Non-Speech):** 使用 `DNS-Challenge` 中的 `noise/` 目录下的纯噪声文件，以及/或者 `LibriSpeech` 音频文件中的静音片段（如果存在）。
        3.  **合成策略：** 创建混合样本。例如，从LibriSpeech中取一段语音，再从DNS-Challenge的noise中取一段噪声，将它们混合（并/或拼接），同时生成对应的逐帧VAD标签（Label）。
              * *例如：[Noise] + [Speech] + [Noise] -\> 对应的Label [0, 0, ..., 1, 1, ..., 0, 0]*
      * **输入：** 合成后的音频。
      * **输出：** 逐帧的二分类标签 (0=Non-Speech, 1=Speech)。

3.  **数据加载器 (DataLoader):**

      * 实现一个高效的数据加载器，能够动态地、按需地从上述两个任务中采样数据（或混合数据），并进行必要的特征提取（如STFT）。

#### 任务3：模型设计与实现

1.  **模块化实现 (Modularity):**
      * **要求：** 你的代码实现必须具有**高度模块化**和**易扩展性**。
      * **设计：** 应该清晰地分离“特征提取器”、“共享主干网络 (Backbone)”和“任务头 (Task Heads)”。这样便于我们未来替换或添加新的任务头（如语音分离、噪声抑制等）。
2.  **多任务损失函数 (Multi-Task Loss):**
      * 实现一个灵活的损失函数，用于联合优化两个任务。
      * $L_{\text{total}} = \lambda_{\text{AEC}} \cdot L_{\text{AEC}} + \lambda_{\text{VAD}} \cdot L_{\text{VAD}}$
      * $L_{\text{AEC}}$：AEC的损失，可以是时域损失（如 SI-SNR, SI-SDR）或频域损失（如幅度(Magnitude)和相位的MSE）。
      * $L_{\text{VAD}}$：VAD的损失，通常为逐帧的二元交叉熵 (Binary Cross-Entropy)。
      * $\lambda$ 是可调的权重超参数。

#### 任务4：实验与评估

1.  **训练脚本：** 提供完整的模型训练和验证脚本。
2.  **评估指标：**
      * **AEC:** PESQ, STOI, SI-SDR, ERLE。
      * **VAD:** 准确率 (Accuracy), 精确率 (Precision), 召回率 (Recall), F1-Score。
3.  **对比分析：** 对你所提出的至少两种模型架构进行实验对比，分析它们在两个任务上的性能表现、计算复杂度及收敛情况。

### 3\. 💾 数据集目录结构参考

**重要提示：** Agent环境中没有这些文件。这些结构仅供你设计预处理脚本时参考。你的脚本必须能够处理基于此结构的数据。

#### DNS-Challenge

```
/mnt/work/liguoteng/dataset/tmp/DNS-Challenge/datasets
├── clean
│   ├── emotional_speech
│   └── read_speech
├── dev_testset
├── impulse_responses
│   ├── SLR26
│   └── SLR28
└── noise

(8 directories)
```

#### LibriSpeech

```
/mnt/nfs/working/acoustic/VPR/opensource_data/LibriSpeech/LibriSpeech
├── BOOKS.TXT
├── CHAPTERS.TXT
├── LICENSE.TXT
├── README.TXT
├── SPEAKERS.TXT
├── dev-clean
├── dev-other
├── test-clean
├── test-other
├── train-clean-100
├── train-clean-360
└── train-other-500
```

### 4\. 🔑 关键交付物 (Deliverables)

1.  **代码：**
      * `preprocessing/`: 数据预处理和合成的完整Python脚本。
      * `models/`: 模块化实现的（至少两个）多任务模型架构。
      * `train.py`: 训练、验证和评估的主脚本。
      * `requirements.txt`: 依赖库列表。
2.  **单元测试 (Unit Tests):**
      * **要求：** 鉴于你无法访问真实数据，请为数据预处理和模型构建的核心逻辑编写单元测试。
      * **策略：** 使用**mock数据**或**自动生成的小型Numpy/Torch张量**作为测试输入。
      * **禁止：** **不要**将任何大型的“假测试数据文件”提交到代码仓库。
3.  **文档 (Documentation):**
      * 一份 `README.md` 文件，清晰说明：
          * 所选模型架构的设计思路和对比。
          * 数据预处理流程。
          * 如何运行预处理、训练和评估。
          * （若有）实验结果总结。

-----

请从\*\*任务1（文献调研）**和**任务2（数据预处理策略和脚本框架）\*\*开始着手。
