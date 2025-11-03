"""Data preprocessing for Acoustic Echo Cancellation (AEC) task."""

import os
import random
import numpy as np
import soundfile as sf
from typing import Tuple, Optional, List
from pathlib import Path

from utils.audio_utils import apply_rir, mix_signals, normalize_audio


class AECDataPreprocessor:
    """
    Preprocessor for AEC data based on DNS-Challenge dataset.
    
    Generates (near-end speech, far-end speech, microphone) data pairs.
    """
    
    def __init__(self,
                 dns_root: str,
                 sample_rate: int = 16000,
                 ser_range: Tuple[float, float] = (-5.0, 15.0),
                 snr_range: Tuple[float, float] = (0.0, 30.0),
                 duration: float = 4.0):
        """
        Initialize AEC data preprocessor.
        
        Args:
            dns_root: Root directory of DNS-Challenge dataset
            sample_rate: Target sample rate in Hz
            ser_range: Range of Signal-to-Echo Ratio in dB (min, max)
            snr_range: Range of Signal-to-Noise Ratio in dB (min, max)
            duration: Duration of generated samples in seconds
        """
        self.dns_root = Path(dns_root)
        self.sample_rate = sample_rate
        self.ser_range = ser_range
        self.snr_range = snr_range
        self.duration = duration
        self.target_samples = int(duration * sample_rate)
        
        # Define paths
        self.clean_speech_dirs = [
            self.dns_root / 'clean' / 'read_speech',
            self.dns_root / 'clean' / 'emotional_speech',
        ]
        self.rir_dirs = [
            self.dns_root / 'impulse_responses' / 'SLR26',
            self.dns_root / 'impulse_responses' / 'SLR28',
        ]
        self.noise_dir = self.dns_root / 'noise'
        
        # Cache file lists
        self.clean_files: List[Path] = []
        self.rir_files: List[Path] = []
        self.noise_files: List[Path] = []
        
    def scan_files(self):
        """Scan and cache audio file paths."""
        # Scan clean speech files
        for clean_dir in self.clean_speech_dirs:
            if clean_dir.exists():
                self.clean_files.extend(
                    list(clean_dir.rglob('*.wav')) + 
                    list(clean_dir.rglob('*.flac'))
                )
        
        # Scan RIR files
        for rir_dir in self.rir_dirs:
            if rir_dir.exists():
                self.rir_files.extend(
                    list(rir_dir.rglob('*.wav')) + 
                    list(rir_dir.rglob('*.flac'))
                )
        
        # Scan noise files
        if self.noise_dir.exists():
            self.noise_files.extend(
                list(self.noise_dir.rglob('*.wav')) + 
                list(self.noise_dir.rglob('*.flac'))
            )
        
        print(f"Found {len(self.clean_files)} clean speech files")
        print(f"Found {len(self.rir_files)} RIR files")
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
        
        # Resample if necessary (simple approach, for production use librosa.resample)
        if sr != self.sample_rate:
            # Simple linear interpolation resampling
            ratio = self.sample_rate / sr
            new_length = int(len(signal) * ratio)
            signal = np.interp(
                np.linspace(0, len(signal) - 1, new_length),
                np.arange(len(signal)),
                signal
            )
        
        # Adjust length
        if len(signal) < target_length:
            # Pad if too short
            signal = np.pad(signal, (0, target_length - len(signal)))
        elif len(signal) > target_length:
            # Random crop if too long
            start = random.randint(0, len(signal) - target_length)
            signal = signal[start:start + target_length]
        
        return signal
    
    def generate_sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate one AEC training sample.
        
        Returns:
            Tuple of (microphone_signal, far_end_signal, near_end_target)
            - microphone_signal: y(t) = s(t) + e(t) + n(t)
            - far_end_signal: x(t)
            - near_end_target: s(t)
        """
        # Check if files are scanned
        if not self.clean_files:
            raise ValueError("No files found. Call scan_files() first.")
        
        # 1. Select near-end speech (target)
        near_end_file = random.choice(self.clean_files)
        near_end = self.load_audio(near_end_file)
        near_end = normalize_audio(near_end)
        
        # 2. Select far-end speech (different from near-end)
        far_end_file = random.choice(self.clean_files)
        while far_end_file == near_end_file and len(self.clean_files) > 1:
            far_end_file = random.choice(self.clean_files)
        far_end = self.load_audio(far_end_file)
        far_end = normalize_audio(far_end)
        
        # 3. Select and apply RIR to generate echo
        if self.rir_files:
            rir_file = random.choice(self.rir_files)
            rir = self.load_audio(rir_file, target_length=None)
            # Normalize RIR
            rir = rir / (np.abs(rir).max() + 1e-8)
            echo = apply_rir(far_end, rir)
        else:
            # If no RIR files, use scaled far-end as echo
            echo = far_end * 0.5
        
        # 4. Select noise
        if self.noise_files:
            noise_file = random.choice(self.noise_files)
            noise = self.load_audio(noise_file)
            noise = normalize_audio(noise)
        else:
            # If no noise files, use zero noise
            noise = np.zeros_like(near_end)
        
        # 5. Mix signals with random SER and SNR
        ser_db = random.uniform(*self.ser_range)
        snr_db = random.uniform(*self.snr_range)
        
        microphone = mix_signals(near_end, echo, noise, ser_db=ser_db, snr_db=snr_db)
        
        return microphone, far_end, near_end
    
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
            mic, far_end, near_end = self.generate_sample()
            
            # Save to files
            sample_dir = output_path / f'sample_{i:06d}'
            sample_dir.mkdir(exist_ok=True)
            
            sf.write(sample_dir / 'microphone.wav', mic, self.sample_rate)
            sf.write(sample_dir / 'far_end.wav', far_end, self.sample_rate)
            sf.write(sample_dir / 'near_end.wav', near_end, self.sample_rate)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
        
        print(f"Completed {split} set generation.")
