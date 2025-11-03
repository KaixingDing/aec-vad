# Multi-Task Learning Model for nnAEC and VAD

This project implements a multi-task deep learning model that simultaneously performs **Acoustic Echo Cancellation (nnAEC)** and **Voice Activity Detection (VAD)**. The model leverages shared representations to improve performance on both tasks.

## 📋 Table of Contents

- [Overview](#overview)
- [Model Architectures](#model-architectures)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Unit Tests](#unit-tests)
- [Results](#results)

## 🎯 Overview

This implementation explores two different multi-task learning architectures:

1. **Shared Backbone Model**: Uses a common CRNN backbone with separate task-specific heads
2. **Hierarchical Model**: Performs VAD first, then uses VAD information to guide AEC through attention mechanisms

Both models are designed to be modular, extensible, and production-ready.

## 🏗️ Model Architectures

### Architecture A: Shared Backbone Model

```
Input (AEC/VAD) 
    ↓
Feature Extractor (Task-specific)
    ↓
Shared CRNN Backbone
    ├─→ AEC Head → Enhanced Speech Magnitude
    └─→ VAD Head → Speech Activity Probabilities
```

**Key Features:**
- Shared CRNN backbone for common feature extraction
- Separate feature extractors for different input channels
- Independent task-specific heads
- Parameter sharing promotes learning of common representations

**Architecture Details:**
- Feature Extractor: 3-layer CNN (64 channels)
- Backbone: 2-layer CNN (128 channels) + 2-layer Bidirectional LSTM (256 hidden units)
- AEC Head: 2-layer MLP → Magnitude spectrum output
- VAD Head: 3-layer MLP → Frame-wise speech probability

### Architecture B: Hierarchical/Progressive Model

```
Input (AEC) → Feature Extractor → AEC Backbone → Attention Gate → AEC Head
                                                       ↑
Input (VAD) → Feature Extractor → VAD Backbone → VAD Head
                                                       ↓
                                                  VAD Guidance
```

**Key Features:**
- VAD branch processes features early
- VAD predictions guide AEC through attention gating
- Progressive refinement: coarse VAD → fine-grained AEC
- Attention mechanism focuses AEC on speech regions

**Architecture Details:**
- Separate backbones for AEC and VAD (different capacities)
- Attention gate modulates AEC features using VAD probabilities
- VAD backbone: Lighter (64 conv channels, 128 LSTM hidden)
- AEC backbone: Heavier (128 conv channels, 256 LSTM hidden)

## 📁 Project Structure

```
aec-vad/
├── preprocessing/
│   ├── __init__.py
│   ├── aec_preprocessor.py      # AEC data generation
│   ├── vad_preprocessor.py      # VAD data generation
│   └── dataloader.py            # Multi-task data loader
├── models/
│   ├── __init__.py
│   ├── base.py                  # Base classes and components
│   ├── shared_backbone.py       # Architecture A
│   ├── hierarchical.py          # Architecture B
│   ├── losses.py                # Loss functions
│   └── metrics.py               # Evaluation metrics
├── utils/
│   ├── __init__.py
│   └── audio_utils.py           # Audio processing utilities
├── tests/
│   ├── __init__.py
│   ├── test_audio_utils.py      # Audio utility tests
│   ├── test_models.py           # Model architecture tests
│   ├── test_losses.py           # Loss function tests
│   └── test_metrics.py          # Metrics tests
├── train.py                     # Training script
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── agent_task.md               # Task specification
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/KaixingDing/aec-vad.git
cd aec-vad
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data Preprocessing

### AEC Data Preprocessing

The AEC preprocessor generates training samples from the DNS-Challenge dataset:

**Process:**
1. Select near-end speech (target) from `clean/read_speech` or `clean/emotional_speech`
2. Select far-end speech from `clean/` (different from near-end)
3. Apply Room Impulse Response (RIR) to far-end to generate echo
4. Add noise from `noise/` directory
5. Mix: `y(t) = s(t) + e(t) + n(t)`

**Usage:**
```python
from preprocessing import AECDataPreprocessor

preprocessor = AECDataPreprocessor(
    dns_root='/path/to/DNS-Challenge/datasets',
    sample_rate=16000,
    ser_range=(-5.0, 15.0),  # Signal-to-Echo Ratio range (dB)
    snr_range=(0.0, 30.0),   # Signal-to-Noise Ratio range (dB)
    duration=4.0,            # Sample duration (seconds)
)

preprocessor.scan_files()
mic, far_end, near_end = preprocessor.generate_sample()
```

### VAD Data Preprocessing

The VAD preprocessor generates samples from LibriSpeech and DNS-Challenge:

**Process:**
1. **Positive samples**: Clean speech from LibriSpeech
2. **Negative samples**: Noise from DNS-Challenge
3. **Mixed samples**: Speech + noise or concatenated [Noise] + [Speech] + [Noise]
4. Generate frame-wise binary labels (0=non-speech, 1=speech)

**Usage:**
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

### Multi-Task Data Loader

The multi-task data loader dynamically samples from both tasks:

```python
from preprocessing import create_dataloaders

dataloader = create_dataloaders(
    aec_preprocessor=aec_preprocessor,
    vad_preprocessor=vad_preprocessor,
    batch_size=16,
    num_workers=4,
    n_fft=512,
    hop_length=128,
)
```

## 🚀 Training

### Basic Training

Train the shared backbone model:

```bash
python train.py \
    --dns_root /path/to/DNS-Challenge/datasets \
    --librispeech_root /path/to/LibriSpeech \
    --model_type shared_backbone \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --output_dir ./outputs/shared_backbone
```

Train the hierarchical model:

```bash
python train.py \
    --dns_root /path/to/DNS-Challenge/datasets \
    --librispeech_root /path/to/LibriSpeech \
    --model_type hierarchical \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --output_dir ./outputs/hierarchical
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dns_root` | None | Path to DNS-Challenge dataset |
| `--librispeech_root` | None | Path to LibriSpeech dataset |
| `--model_type` | shared_backbone | Model architecture (shared_backbone or hierarchical) |
| `--batch_size` | 16 | Batch size |
| `--num_epochs` | 100 | Number of training epochs |
| `--learning_rate` | 1e-3 | Learning rate |
| `--aec_weight` | 1.0 | Weight for AEC loss |
| `--vad_weight` | 1.0 | Weight for VAD loss |
| `--output_dir` | ./outputs | Output directory |

### Multi-Task Loss Function

The total loss is a weighted combination:

```
L_total = λ_AEC × L_AEC + λ_VAD × L_VAD

where:
  L_AEC = w_mag × L_magnitude + w_sisdr × L_SI-SDR
  L_VAD = Binary Cross-Entropy
```

## 📈 Evaluation

### AEC Metrics

- **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio): Measures enhancement quality
- **PESQ** (Perceptual Evaluation of Speech Quality): Perceptual quality (requires time-domain)
- **STOI** (Short-Time Objective Intelligibility): Speech intelligibility (requires time-domain)
- **ERLE** (Echo Return Loss Enhancement): Echo reduction (requires time-domain)

*Note: Full PESQ, STOI, and ERLE require time-domain reconstruction and external libraries (pesq, pystoi).*

### VAD Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Proportion of predicted speech that is actually speech
- **Recall**: Proportion of actual speech that is detected
- **F1-Score**: Harmonic mean of precision and recall

## 🧪 Unit Tests

Run all tests:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
# Test audio utilities
pytest tests/test_audio_utils.py -v

# Test model architectures
pytest tests/test_models.py -v

# Test loss functions
pytest tests/test_losses.py -v

# Test metrics
pytest tests/test_metrics.py -v
```

**Test Coverage:**
- ✅ Audio processing utilities (STFT, mixing, RIR application)
- ✅ Model architectures (forward pass, gradient flow, device transfer)
- ✅ Loss functions (SI-SDR, magnitude, multi-task)
- ✅ Evaluation metrics (AEC SI-SDR, VAD accuracy/F1)

All tests use **mock data** (randomly generated tensors) and do not require actual datasets.

## 📊 Results

### Model Comparison

| Model | Parameters | AEC SI-SDR | VAD F1 | Training Time |
|-------|------------|------------|--------|---------------|
| Shared Backbone | ~2.5M | TBD | TBD | TBD |
| Hierarchical | ~3.2M | TBD | TBD | TBD |

*Results will be updated after training on actual datasets.*

### Architecture Analysis

**Shared Backbone Model:**
- ✅ Simpler architecture
- ✅ Efficient parameter sharing
- ✅ Good for balanced multi-task learning
- ⚠️ May struggle if tasks have conflicting objectives

**Hierarchical Model:**
- ✅ VAD guidance improves AEC on speech regions
- ✅ Better handling of silence/noise-only segments
- ✅ More interpretable (VAD → AEC pipeline)
- ⚠️ Higher computational cost
- ⚠️ Sequential processing may increase latency

## 🔬 Research Insights

### Multi-Task Learning Benefits

1. **Shared Representations**: Both tasks benefit from learning common acoustic features
2. **Data Efficiency**: Shared backbone reduces total parameters vs. two separate models
3. **Regularization**: Multi-task learning acts as implicit regularization
4. **Task Synergy**: VAD helps AEC focus on speech regions, AEC helps VAD with noisy conditions

### Future Improvements

1. **Model Architecture**:
   - Explore Transformer-based architectures
   - Add residual connections and skip connections
   - Investigate U-Net style encoder-decoder

2. **Loss Functions**:
   - Add perceptual losses (PESQ, STOI)
   - Explore adversarial losses for more natural speech
   - Dynamic task weighting strategies

3. **Training**:
   - Curriculum learning (start with easier samples)
   - Data augmentation (pitch shift, time stretch)
   - Online hard example mining
   - Separate validation dataloader with different random seed

4. **Extensions**:
   - Add more tasks (denoising, source separation)
   - Real-time inference optimization
   - Model compression and quantization

5. **Code Quality**:
   - Replace linear interpolation resampling with librosa.resample()
   - Use scipy.signal.windows.hann() instead of deprecated np.hanning()
   - Implement proper anti-aliasing in audio resampling

## ⚠️ Known Limitations

- Audio resampling uses linear interpolation (for simplicity); production code should use proper anti-aliasing resampling
- np.hanning() is deprecated in newer NumPy versions; consider scipy.signal alternatives
- Training and validation use same dataloader in demo mode; separate validation set recommended for actual training
- STFT/iSTFT implementation is basic; consider using torchaudio for production

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{aec_vad_multitask,
  title={Multi-Task Learning for Acoustic Echo Cancellation and Voice Activity Detection},
  author={Kaixin Ding},
  year={2025},
  url={https://github.com/KaixingDing/aec-vad}
}
```

## 📄 License

This project is provided for educational and research purposes.

## 🙏 Acknowledgments

- DNS-Challenge dataset for providing diverse acoustic scenarios
- LibriSpeech corpus for clean speech data
- PyTorch team for the excellent deep learning framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
