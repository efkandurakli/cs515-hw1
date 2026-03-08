"""
Training module for MNIST MLP classification.

This module implements the training loop with support for various optimizers,
learning rate schedulers, regularization, early stopping, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import os
import json

from parameters import TrainingConfig


def get_optimizer(
    model: nn.Module,
    config: TrainingConfig
) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: PyTorch model to optimize.
        config: Training configuration containing optimizer settings.
    
    Returns:
        Configured optimizer instance.
    """
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    return optimizer


def get_scheduler(
    optimizer: optim.Optimizer,
    config: TrainingConfig
) -> Optional[optim.lr_scheduler.LRScheduler]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule.
        config: Training configuration containing scheduler settings.
    
    Returns:
        Configured scheduler instance or None.
    """
    if config.scheduler is None:
        return None
    elif config.scheduler == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )
    elif config.scheduler == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.scheduler_patience,
            factor=config.scheduler_gamma
        )
    elif config.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.lr * 0.01
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


def compute_l1_loss(model: nn.Module) -> torch.Tensor:
    """
    Compute L1 regularization loss for model parameters.
    
    Args:
        model: PyTorch model.
    
    Returns:
        L1 loss tensor.
    """
    l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return l1_loss


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    config: TrainingConfig,
    log_interval: int = 100
) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on ('cuda' or 'cpu').
        config: Training configuration.
        log_interval: Log interval in batches.
    
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        if config.l1_lambda > 0:
            l1_loss = compute_l1_loss(model)
            loss = loss + config.l1_lambda * l1_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """
    Validate model on validation set.
    
    Args:
        model: PyTorch model to validate.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Device to validate on ('cuda' or 'cpu').
    
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validating', leave=False):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: str = 'cpu',
    log_interval: int = 100
) -> Dict:
    """
    Train model with specified configuration.
    
    This function implements the complete training loop including:
    - Training and validation
    - Early stopping based on validation loss
    - Model checkpointing (saving best model)
    - Learning rate scheduling
    - Training history tracking
    
    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        device: Device to train on ('cuda' or 'cpu').
        log_interval: Log interval in batches.
    
    Returns:
        Dictionary containing training history with keys:
        - 'train_losses': List of training losses per epoch
        - 'train_accs': List of training accuracies per epoch
        - 'val_losses': List of validation losses per epoch
        - 'val_accs': List of validation accuracies per epoch
        - 'best_epoch': Epoch with best validation loss
        - 'best_val_loss': Best validation loss achieved
        - 'best_val_acc': Validation accuracy at best epoch
    """
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    history = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
    
    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Optimizer: {config.optimizer}, LR: {config.lr}")
    print(f"Scheduler: {config.scheduler}")
    print(f"Early stopping patience: {config.early_stop_patience}")
    
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, config, log_interval
        )
        
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['val_losses'].append(val_loss)
        history['val_accs'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, config.checkpoint_path)
            print(f"Saved best model to {config.checkpoint_path}")
        else:
            patience_counter += 1
        
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        if patience_counter >= config.early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
            break
    
    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val_loss
    history['best_val_acc'] = history['val_accs'][best_epoch - 1]
    
    history_path = config.checkpoint_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {history_path}")
    
    return history
