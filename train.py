"""Training script for multi-task AEC-VAD model."""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from models import SharedBackboneModel, HierarchicalModel
from models.losses import MultiTaskLoss
from models.metrics import MultiTaskMetrics
from preprocessing import AECDataPreprocessor, VADDataPreprocessor, create_dataloaders


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Args:
        model: Multi-task model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Dictionary with average losses
    """
    model.train()
    
    total_losses = {}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        optimizer.zero_grad()
        
        outputs = {}
        targets = {}
        
        # Process AEC batch if present
        if 'aec' in batch:
            aec_input = batch['aec']['input'].to(device)
            aec_target = batch['aec']['target_mag'].to(device)
            
            aec_output = model(aec_input, task='aec')
            outputs['aec'] = aec_output
            targets['aec'] = {'target_mag': aec_target}
        
        # Process VAD batch if present
        if 'vad' in batch:
            vad_input = batch['vad']['input'].to(device)
            vad_target = batch['vad']['target_labels'].to(device)
            
            vad_output = model(vad_input, task='vad')
            outputs['vad'] = vad_output
            targets['vad'] = {'target_labels': vad_target}
        
        # Compute loss
        losses = criterion(outputs, targets)
        total_loss = losses['total']
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        for key, value in losses.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            total_losses[key] += value.item()
        
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': total_loss.item()})
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses


def validate(model, dataloader, criterion, metrics, device):
    """
    Validate the model.
    
    Args:
        model: Multi-task model
        dataloader: Validation data loader
        criterion: Loss function
        metrics: Metrics tracker
        device: Device to validate on
    
    Returns:
        Tuple of (average losses, computed metrics)
    """
    model.eval()
    metrics.reset()
    
    total_losses = {}
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        
        for batch in pbar:
            outputs = {}
            targets = {}
            
            # Process AEC batch if present
            if 'aec' in batch:
                aec_input = batch['aec']['input'].to(device)
                aec_target = batch['aec']['target_mag'].to(device)
                
                aec_output = model(aec_input, task='aec')
                outputs['aec'] = aec_output
                targets['aec'] = {'target_mag': aec_target}
            
            # Process VAD batch if present
            if 'vad' in batch:
                vad_input = batch['vad']['input'].to(device)
                vad_target = batch['vad']['target_labels'].to(device)
                
                vad_output = model(vad_input, task='vad')
                outputs['vad'] = vad_output
                targets['vad'] = {'target_labels': vad_target}
            
            # Compute loss
            losses = criterion(outputs, targets)
            
            # Update metrics
            metrics.update(outputs, targets)
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()
            
            num_batches += 1
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    # Compute metrics
    computed_metrics = metrics.compute()
    
    return avg_losses, computed_metrics


def main(args):
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    # Create preprocessors (mock mode for when datasets are not available)
    aec_preprocessor = None
    vad_preprocessor = None
    
    if args.dns_root and Path(args.dns_root).exists():
        aec_preprocessor = AECDataPreprocessor(
            dns_root=args.dns_root,
            sample_rate=args.sample_rate,
            duration=args.duration,
        )
        aec_preprocessor.scan_files()
    
    if (args.librispeech_root and Path(args.librispeech_root).exists()) or \
       (args.dns_root and Path(args.dns_root).exists()):
        vad_preprocessor = VADDataPreprocessor(
            librispeech_root=args.librispeech_root,
            dns_root=args.dns_root,
            sample_rate=args.sample_rate,
            duration=args.duration,
        )
        vad_preprocessor.scan_files()
    
    # Create data loaders
    if aec_preprocessor or vad_preprocessor:
        train_loader = create_dataloaders(
            aec_preprocessor=aec_preprocessor,
            vad_preprocessor=vad_preprocessor,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )
        val_loader = train_loader  # In practice, should be separate
    else:
        print("Warning: No datasets found. Training will not be performed.")
        train_loader = None
        val_loader = None
    
    # Create model
    if args.model_type == 'shared_backbone':
        model = SharedBackboneModel(
            n_freqs=args.n_fft // 2 + 1,
            aec_channels=2,
            vad_channels=1,
        )
    elif args.model_type == 'hierarchical':
        model = HierarchicalModel(
            n_freqs=args.n_fft // 2 + 1,
            aec_channels=2,
            vad_channels=1,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"\nModel: {args.model_type}")
    print(f"Total parameters: {model_info['total']:,}")
    for key, value in model_info.items():
        if key != 'total':
            print(f"  {key}: {value:,}")
    
    # Create loss function
    criterion = MultiTaskLoss(
        aec_weight=args.aec_weight,
        vad_weight=args.vad_weight,
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Create metrics tracker
    metrics = MultiTaskMetrics()
    
    # Training loop
    best_val_loss = float('inf')
    
    if train_loader is not None:
        for epoch in range(1, args.num_epochs + 1):
            # Train
            train_losses = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # Validate
            val_losses, val_metrics = validate(
                model, val_loader, criterion, metrics, device
            )
            
            # Log to tensorboard
            for key, value in train_losses.items():
                writer.add_scalar(f'Train/{key}', value, epoch)
            
            for key, value in val_losses.items():
                writer.add_scalar(f'Val/{key}', value, epoch)
            
            for task, task_metrics in val_metrics.items():
                for metric_name, metric_value in task_metrics.items():
                    writer.add_scalar(f'Val/{task}_{metric_name}', metric_value, epoch)
            
            # Print results
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            
            if 'aec' in val_metrics:
                print(f"  AEC SI-SDR: {val_metrics['aec']['si_sdr']:.2f} dB")
            
            if 'vad' in val_metrics:
                print(f"  VAD F1: {val_metrics['vad']['f1_score']:.4f}")
                print(f"  VAD Accuracy: {val_metrics['vad']['accuracy']:.4f}")
            
            # Update learning rate
            scheduler.step(val_losses['total'])
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_losses['total'],
                    'val_metrics': val_metrics,
                }, output_dir / 'best_model.pt')
                print(f"  Saved best model (val_loss: {val_losses['total']:.4f})")
            
            # Save checkpoint
            if epoch % args.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    else:
        # Save untrained model for testing purposes
        torch.save({
            'epoch': 0,
            'model_state_dict': model.state_dict(),
        }, output_dir / 'model_structure.pt')
        print("\nNo training performed (datasets not available).")
        print(f"Model structure saved to {output_dir / 'model_structure.pt'}")
    
    writer.close()
    print("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multi-task AEC-VAD model')
    
    # Data arguments
    parser.add_argument('--dns_root', type=str, default=None,
                        help='Path to DNS-Challenge dataset')
    parser.add_argument('--librispeech_root', type=str, default=None,
                        help='Path to LibriSpeech dataset')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='shared_backbone',
                        choices=['shared_backbone', 'hierarchical'],
                        help='Model architecture type')
    
    # Audio arguments
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--duration', type=float, default=4.0,
                        help='Audio duration in seconds')
    parser.add_argument('--n_fft', type=int, default=512,
                        help='FFT size for STFT')
    parser.add_argument('--hop_length', type=int, default=128,
                        help='Hop length for STFT')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--aec_weight', type=float, default=1.0,
                        help='Weight for AEC loss')
    parser.add_argument('--vad_weight', type=float, default=1.0,
                        help='Weight for VAD loss')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')
    
    args = parser.parse_args()
    
    main(args)
