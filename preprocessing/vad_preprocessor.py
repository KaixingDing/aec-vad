"""Data preprocessing for Voice Activity Detection (VAD) task."""

import os
import random
import numpy as np
import soundfile as sf
from typing import Tuple, Optional, List
from pathlib import Path

from utils.audio_utils import generate_vad_labels, normalize_audio


class VADDataPreprocessor:
    """
    Preprocessor for VAD data based on LibriSpeech and DNS-Challenge datasets.
    
    Generates (mixed audio, frame-wise VAD labels) data pairs.
    """
    
    def __init__(self,
                 librispeech_root: Optional[str] = None,
                 dns_root: Optional[str] = None,
                 sample_rate: int = 16000,
                 snr_range: Tuple[float, float] = (0.0, 30.0),
                 duration: float = 4.0,
                 frame_length: float = 0.025,
                 hop_length: float = 0.010):
        """
        Initialize VAD data preprocessor.
        
        Args:
            librispeech_root: Root directory of LibriSpeech dataset
            dns_root: Root directory of DNS-Challenge dataset
            sample_rate: Target sample rate in Hz
            snr_range: Range of Signal-to-Noise Ratio in dB (min, max)
            duration: Duration of generated samples in seconds
            frame_length: Frame length in seconds for VAD labels
            hop_length: Hop length in seconds for VAD labels
        """
        self.librispeech_root = Path(librispeech_root) if librispeech_root else None
        self.dns_root = Path(dns_root) if dns_root else None
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.duration = duration
        self.target_samples = int(duration * sample_rate)
        self.frame_length = frame_length
        self.hop_length = hop_length
        
        # Cache file lists
        self.speech_files: List[Path] = []
        self.noise_files: List[Path] = []
        
    def scan_files(self):
        """Scan and cache audio file paths."""
        # Scan LibriSpeech files
        if self.librispeech_root and self.librispeech_root.exists():
            for split in ['train-clean-100', 'train-clean-360', 'dev-clean']:
                split_dir = self.librispeech_root / split
                if split_dir.exists():
                    self.speech_files.extend(list(split_dir.rglob('*.flac')))
        
        # Scan DNS-Challenge noise files
        if self.dns_root and self.dns_root.exists():
            noise_dir = self.dns_root / 'noise'
            if noise_dir.exists():
                self.noise_files.extend(
                    list(noise_dir.rglob('*.wav')) + 
                    list(noise_dir.rglob('*.flac'))
                )
        
        print(f"Found {len(self.speech_files)} speech files")
        print(f"Found {len(self.noise_files)} noise files")
    
    def load_audio(self, 
                   file_path: Path, 
                   target_length: Optional[int] = None) -> np.ndarray:
        """
        Load audio file and resample if necessary.
        
        Args:
            file_path: Path to audio file
            target_length: Target length in samples. If None, use self.target_samples
        
        Returns:
            Audio signal as numpy array
        """
        if target_length is None:
            target_length = self.target_samples
        
        # Load audio
        signal, sr = sf.read(str(file_path))
        
        # Convert to mono if stereo
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        
        # Resample if necessary
        if sr != self.sample_rate:
            ratio = self.sample_rate / sr
            new_length = int(len(signal) * ratio)
            signal = np.interp(
                np.linspace(0, len(signal) - 1, new_length),
                np.arange(len(signal)),
                signal
            )
        
        # Adjust length
        if len(signal) < target_length:
            signal = np.pad(signal, (0, target_length - len(signal)))
        elif len(signal) > target_length:
            start = random.randint(0, len(signal) - target_length)
            signal = signal[start:start + target_length]
        
        return signal
    
    def generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one VAD training sample with mixed speech and noise.
        
        Returns:
            Tuple of (mixed_audio, vad_labels)
            - mixed_audio: Mixed signal with speech and/or noise segments
            - vad_labels: Frame-wise binary labels (0=non-speech, 1=speech)
        """
        # Decide composition: speech only, noise only, or mixed
        composition = random.choice(['speech', 'noise', 'mixed', 'concatenated'])
        
        if composition == 'speech' and self.speech_files:
            # Pure speech
            signal = self.load_speech_segment()
            labels = np.ones(self.get_num_frames(), dtype=np.int32)
            
        elif composition == 'noise' and self.noise_files:
            # Pure noise
            signal = self.load_noise_segment()
            labels = np.zeros(self.get_num_frames(), dtype=np.int32)
            
        elif composition == 'mixed' and self.speech_files and self.noise_files:
            # Speech + noise overlay
            speech = self.load_speech_segment()
            noise = self.load_noise_segment()
            
            # Mix with random SNR
            snr_db = random.uniform(*self.snr_range)
            speech_power = np.mean(speech ** 2)
            noise_power = np.mean(noise ** 2)
            
            if noise_power > 0:
                noise_scale = np.sqrt(speech_power / (noise_power * (10 ** (snr_db / 10))))
                noise = noise * noise_scale
            
            signal = speech + noise
            
            # Normalize
            max_val = np.abs(signal).max()
            if max_val > 0.95:
                signal = signal * 0.95 / max_val
            
            # All frames are speech
            labels = np.ones(self.get_num_frames(), dtype=np.int32)
            
        elif composition == 'concatenated' and self.speech_files and self.noise_files:
            # Concatenate noise + speech + noise segments
            signal, labels = self.generate_concatenated_sample()
            
        else:
            # Fallback: generate silence
            signal = np.zeros(self.target_samples, dtype=np.float32)
            labels = np.zeros(self.get_num_frames(), dtype=np.int32)
        
        return signal, labels
    
    def load_speech_segment(self) -> np.ndarray:
        """Load a speech segment."""
        if not self.speech_files:
            return np.zeros(self.target_samples, dtype=np.float32)
        
        speech_file = random.choice(self.speech_files)
        speech = self.load_audio(speech_file)
        return normalize_audio(speech)
    
    def load_noise_segment(self) -> np.ndarray:
        """Load a noise segment."""
        if not self.noise_files:
            return np.random.randn(self.target_samples) * 0.01
        
        noise_file = random.choice(self.noise_files)
        noise = self.load_audio(noise_file)
        return normalize_audio(noise)
    
    def generate_concatenated_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a sample with concatenated segments: [Noise] + [Speech] + [Noise].
        
        Returns:
            Tuple of (signal, labels)
        """
        # Randomly decide segment lengths (in samples)
        noise1_len = random.randint(
            int(0.5 * self.sample_rate),  # 0.5 seconds
            int(1.5 * self.sample_rate)   # 1.5 seconds
        )
        speech_len = random.randint(
            int(1.0 * self.sample_rate),  # 1.0 seconds
            int(2.5 * self.sample_rate)   # 2.5 seconds
        )
        noise2_len = self.target_samples - noise1_len - speech_len
        
        if noise2_len < 0:
            # Adjust if total exceeds target
            speech_len = self.target_samples - noise1_len - int(0.5 * self.sample_rate)
            noise2_len = self.target_samples - noise1_len - speech_len
        
        # Load segments
        noise1 = self.load_noise_segment()[:noise1_len]
        speech = self.load_speech_segment()[:speech_len]
        noise2 = self.load_noise_segment()[:noise2_len]
        
        # Concatenate
        signal = np.concatenate([noise1, speech, noise2])
        
        # Ensure exact length
        if len(signal) < self.target_samples:
            signal = np.pad(signal, (0, self.target_samples - len(signal)))
        else:
            signal = signal[:self.target_samples]
        
        # Generate labels based on segment positions
        frame_samples = int(self.frame_length * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)
        n_frames = self.get_num_frames()
        labels = np.zeros(n_frames, dtype=np.int32)
        
        # Mark speech frames
        speech_start_sample = noise1_len
        speech_end_sample = noise1_len + speech_len
        
        for i in range(n_frames):
            frame_start = i * hop_samples
            frame_end = frame_start + frame_samples
            
            # Check if frame overlaps with speech segment
            if frame_end > speech_start_sample and frame_start < speech_end_sample:
                labels[i] = 1
        
        return signal, labels
    
    def get_num_frames(self) -> int:
        """Calculate number of frames for the target duration."""
        frame_samples = int(self.frame_length * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)
        return 1 + (self.target_samples - frame_samples) // hop_samples
    
    def preprocess_dataset(self, 
                          num_samples: int,
                          output_dir: str,
                          split: str = 'train'):
        """
        Preprocess and save dataset.
        
        Args:
            num_samples: Number of samples to generate
            output_dir: Output directory
            split: Dataset split name ('train', 'val', 'test')
        """
        output_path = Path(output_dir) / split
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {num_samples} {split} samples...")
        
        for i in range(num_samples):
            signal, labels = self.generate_sample()
            
            # Save to files
            sample_dir = output_path / f'sample_{i:06d}'
            sample_dir.mkdir(exist_ok=True)
            
            sf.write(sample_dir / 'audio.wav', signal, self.sample_rate)
            np.save(sample_dir / 'labels.npy', labels)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
        
        print(f"Completed {split} set generation.")
