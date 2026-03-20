#!/usr/bin/env python
"""
S3T-Former Training Script (Clean Version)
For NTU Skeleton Action Recognition
"""

import argparse
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import yaml
from datetime import datetime
from spikingjelly.activation_based import functional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feeders.feeder_ntus import Feeder


def init_seed(seed=1):
    """Initialize random seed"""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    """Dynamically import class"""
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(f'Class {class_str} cannot be found in {mod_str}')


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None, use_amp=False):
    """Train for one epoch"""
    model.train()
    losses = []
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, label, index) in enumerate(pbar):
        # Data format: (N, C, T, V, M)
        if data.dim() == 4:
            data = data.unsqueeze(-1)  # Add M dimension
        
        data = data.to(device)
        label = label.to(device)
        
        # Reset network state
        functional.reset_net(model)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                output = model(data)
                loss = criterion(output, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
        pred = output.argmax(dim=1)
        correct += pred.eq(label).sum().item()
        total += label.size(0)
        
        if batch_idx % 100 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return np.mean(losses), 100. * correct / total


def test(model, test_loader, criterion, device):
    """Test model"""
    model.eval()
    losses = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, label, index in tqdm(test_loader, desc='Testing'):
            if data.dim() == 4:
                data = data.unsqueeze(-1)
            
            data = data.to(device)
            label = label.to(device)
            
            functional.reset_net(model)
            output = model(data)
            loss = criterion(output, label)
            
            losses.append(loss.item())
            pred = output.argmax(dim=1)
            correct += pred.eq(label).sum().item()
            total += label.size(0)
    
    return np.mean(losses), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='S3T-Former Training')
    parser.add_argument('--config', default='config/nturgbd-cross-subject-s3t-former.yaml', help='Config file path')
    parser.add_argument('--work-dir', default='./logs', help='Work directory')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()
    
    # Initialize random seed
    init_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device_ids = config.get('device', [0])
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model_path = config['model']
    ModelClass = import_class(model_path)
    model = ModelClass(**config['model_args'])
    model = model.to(device)
    
    # Use multi-GPU if available
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params/1e6:.2f}M')
    print(f'Trainable parameters: {trainable_params/1e6:.2f}M')
    
    # Create data loaders
    FeederClass = import_class(config['feeder'])
    train_dataset = FeederClass(**config['train_feeder_args'])
    test_dataset = FeederClass(**config['test_feeder_args'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=True,
        num_workers=config.get('num_worker', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('test_batch_size', 128),
        shuffle=False,
        num_workers=config.get('num_worker', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    # Create optimizer
    optimizer_name = config.get('optimizer', 'AdamW').lower()
    base_lr = config.get('base_lr', 0.01)
    weight_decay = config.get('weight_decay', 0.0005)
    optimizer_args = config.get('optimizer_args', {})
    
    if optimizer_name == 'adamw':
        betas = optimizer_args.get('betas', (0.9, 0.999))
        if isinstance(betas, list):
            betas = tuple(betas)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_args.get('adamw_lr', base_lr * 0.03),
            weight_decay=optimizer_args.get('adamw_weight_decay', weight_decay),
            betas=betas
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    # Create learning rate scheduler
    num_epoch = config.get('num_epoch', 250)
    if config.get('use_cosine_annealing', True):
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=config.get('min_lr', 1e-6))
    else:
        step = config.get('step', [110, 170, 230])
        scheduler = MultiStepLR(optimizer, milestones=step, gamma=config.get('lr_decay_rate', 0.1))
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training
    use_amp = config.get('use_amp', True)
    scaler = GradScaler() if use_amp else None
    
    # Create work directory
    work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True)
    
    # Save config
    config_save_path = os.path.join(work_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    
    # Training loop
    best_acc = 0.0
    num_epoch = config.get('num_epoch', 250)
    eval_interval = config.get('eval_interval', 5)
    
    print(f'\nStarting training for {num_epoch} epochs...')
    print('='*80)
    
    for epoch in range(1, num_epoch + 1):
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler, use_amp
        )
        
        # Update learning rate
        scheduler.step()
        
        # Evaluation
        if epoch % eval_interval == 0 or epoch == num_epoch:
            test_loss, test_acc = test(model, test_loader, criterion, device)
            
            print(f'\nEpoch {epoch}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_model_path = os.path.join(work_dir, 'best_model.pth')
                if isinstance(model, nn.DataParallel):
                    torch.save({'model_state_dict': model.module.state_dict()}, best_model_path)
                else:
                    torch.save({'model_state_dict': model.state_dict()}, best_model_path)
                print(f'  ✓ Saved best model (Acc: {best_acc:.2f}%)')
        
        # Save latest model
        if epoch % config.get('save_interval', 1) == 0:
            latest_model_path = os.path.join(work_dir, 'latest_model.pth')
            if isinstance(model, nn.DataParallel):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                }, latest_model_path)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                }, latest_model_path)
    
    print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()

