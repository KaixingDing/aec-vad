"""Data loading and batching for multi-task learning."""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
import random

from utils.audio_utils import stft_transform


class MultiTaskDataset(Dataset):
    """
    Multi-task dataset that can sample from both AEC and VAD tasks.
    """
    
    def __init__(self,
                 aec_preprocessor=None,
                 vad_preprocessor=None,
                 n_fft: int = 512,
                 hop_length: int = 128,
                 task_balance: float = 0.5):
        """
        Initialize multi-task dataset.
        
        Args:
            aec_preprocessor: AEC data preprocessor instance
            vad_preprocessor: VAD data preprocessor instance
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
            task_balance: Probability of sampling AEC vs VAD (0.5 = equal)
        """
        self.aec_preprocessor = aec_preprocessor
        self.vad_preprocessor = vad_preprocessor
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.task_balance = task_balance
        
        # Validate that at least one preprocessor is provided
        if aec_preprocessor is None and vad_preprocessor is None:
            raise ValueError("At least one preprocessor must be provided")
    
    def __len__(self) -> int:
        """Return a large number since we generate samples on-the-fly."""
        return 10000  # Arbitrary large number for on-the-fly generation
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate one training sample.
        
        Returns:
            Dictionary containing:
            - 'task': 'aec' or 'vad'
            - 'input': Input features (STFT magnitude)
            - 'target': Target output
            - Additional task-specific data
        """
        # Decide which task to sample
        if self.aec_preprocessor and self.vad_preprocessor:
            use_aec = random.random() < self.task_balance
        elif self.aec_preprocessor:
            use_aec = True
        else:
            use_aec = False
        
        if use_aec:
            return self._get_aec_sample()
        else:
            return self._get_vad_sample()
    
    def _get_aec_sample(self) -> Dict[str, torch.Tensor]:
        """Generate an AEC sample."""
        # Generate raw audio
        mic, far_end, near_end = self.aec_preprocessor.generate_sample()
        
        # Compute STFT
        mic_stft = stft_transform(mic, n_fft=self.n_fft, hop_length=self.hop_length)
        far_end_stft = stft_transform(far_end, n_fft=self.n_fft, hop_length=self.hop_length)
        near_end_stft = stft_transform(near_end, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Extract magnitude and phase
        mic_mag = np.abs(mic_stft)
        mic_phase = np.angle(mic_stft)
        far_end_mag = np.abs(far_end_stft)
        near_end_mag = np.abs(near_end_stft)
        near_end_phase = np.angle(near_end_stft)
        
        # Stack microphone and far-end as input (2 channels)
        input_features = np.stack([mic_mag, far_end_mag], axis=0)  # (2, n_freqs, n_frames)
        
        # Convert to torch tensors
        return {
            'task': 'aec',
            'input': torch.FloatTensor(input_features),
            'target_mag': torch.FloatTensor(near_end_mag),
            'target_phase': torch.FloatTensor(near_end_phase),
            'mic_phase': torch.FloatTensor(mic_phase),
        }
    
    def _get_vad_sample(self) -> Dict[str, torch.Tensor]:
        """Generate a VAD sample."""
        # Generate raw audio and labels
        signal, labels = self.vad_preprocessor.generate_sample()
        
        # Compute STFT
        signal_stft = stft_transform(signal, n_fft=self.n_fft, hop_length=self.hop_length)
        signal_mag = np.abs(signal_stft)
        
        # Input is single channel magnitude
        input_features = signal_mag[np.newaxis, :, :]  # (1, n_freqs, n_frames)
        
        # Ensure labels match number of frames
        n_frames = signal_mag.shape[1]
        if len(labels) > n_frames:
            labels = labels[:n_frames]
        elif len(labels) < n_frames:
            labels = np.pad(labels, (0, n_frames - len(labels)))
        
        # Convert to torch tensors
        return {
            'task': 'vad',
            'input': torch.FloatTensor(input_features),
            'target_labels': torch.LongTensor(labels),  # (n_frames,)
        }


def collate_multitask_batch(batch):
    """
    Custom collate function for multi-task batches.
    
    Groups samples by task and creates separate batches.
    """
    # Separate by task
    aec_samples = [s for s in batch if s['task'] == 'aec']
    vad_samples = [s for s in batch if s['task'] == 'vad']
    
    result = {}
    
    # Collate AEC samples
    if aec_samples:
        result['aec'] = {
            'input': torch.stack([s['input'] for s in aec_samples]),
            'target_mag': torch.stack([s['target_mag'] for s in aec_samples]),
            'target_phase': torch.stack([s['target_phase'] for s in aec_samples]),
            'mic_phase': torch.stack([s['mic_phase'] for s in aec_samples]),
        }
    
    # Collate VAD samples
    if vad_samples:
        # Find max frame length for padding
        max_frames = max(s['target_labels'].shape[0] for s in vad_samples)
        
        # Pad inputs and labels to same length
        padded_inputs = []
        padded_labels = []
        
        for s in vad_samples:
            input_tensor = s['input']
            label_tensor = s['target_labels']
            
            # Pad frames dimension
            n_frames = input_tensor.shape[2]
            if n_frames < max_frames:
                pad_size = max_frames - n_frames
                input_tensor = torch.nn.functional.pad(
                    input_tensor, (0, pad_size), mode='constant', value=0
                )
                label_tensor = torch.nn.functional.pad(
                    label_tensor, (0, pad_size), mode='constant', value=0
                )
            
            padded_inputs.append(input_tensor)
            padded_labels.append(label_tensor)
        
        result['vad'] = {
            'input': torch.stack(padded_inputs),
            'target_labels': torch.stack(padded_labels),
        }
    
    return result


def create_dataloaders(aec_preprocessor=None,
                       vad_preprocessor=None,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       n_fft: int = 512,
                       hop_length: int = 128) -> DataLoader:
    """
    Create data loader for multi-task training.
    
    Args:
        aec_preprocessor: AEC preprocessor instance
        vad_preprocessor: VAD preprocessor instance
        batch_size: Batch size
        num_workers: Number of data loading workers
        n_fft: FFT size for STFT
        hop_length: Hop length for STFT
    
    Returns:
        DataLoader instance
    """
    dataset = MultiTaskDataset(
        aec_preprocessor=aec_preprocessor,
        vad_preprocessor=vad_preprocessor,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_multitask_batch,
        pin_memory=True,
    )
    
    return dataloader
