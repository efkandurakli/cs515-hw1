"""
Main entry point for MNIST MLP classification.

This module provides a command-line interface for training and testing
the MLP model on MNIST dataset with various configuration options.

Example usage:
    python main.py --mode train --hidden_dims 256 128 --activation relu --epochs 50
    python main.py --mode test --checkpoint checkpoints/best_model.pth
"""

import argparse
import torch
import random
import numpy as np
import os
import json
from typing import List

from parameters import ModelConfig, TrainingConfig, ExperimentConfig, Config
from models import MLP
from utils.data_loader import get_mnist_loaders
from utils.visualization import (
    plot_training_curves, plot_confusion_matrix,
    plot_tsne, plot_learning_rate_schedule
)
from train import train_model
from test import test_model


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments as Namespace object.
    """
    parser = argparse.ArgumentParser(
        description='MNIST MLP Classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'both'],
        default='train',
        help='Mode: train, test, or both'
    )
    
    parser.add_argument(
        '--hidden_dims',
        type=int,
        nargs='+',
        default=[256, 128],
        help='Hidden layer dimensions'
    )
    parser.add_argument(
        '--activation',
        type=str,
        choices=['relu', 'gelu'],
        default='relu',
        help='Activation function'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='Dropout probability'
    )
    parser.add_argument(
        '--use_bn',
        action='store_true',
        help='Use batch normalization'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['sgd', 'adam', 'adamw'],
        default='adam',
        help='Optimizer type'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for SGD optimizer'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='L2 regularization coefficient'
    )
    parser.add_argument(
        '--l1_lambda',
        type=float,
        default=0.0,
        help='L1 regularization coefficient'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['step', 'plateau', 'cosine'],
        default=None,
        help='Learning rate scheduler'
    )
    parser.add_argument(
        '--scheduler_step_size',
        type=int,
        default=10,
        help='Step size for StepLR scheduler'
    )
    parser.add_argument(
        '--scheduler_gamma',
        type=float,
        default=0.1,
        help='Gamma for learning rate scheduler'
    )
    parser.add_argument(
        '--scheduler_patience',
        type=int,
        default=5,
        help='Patience for ReduceLROnPlateau scheduler'
    )
    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'mps', 'auto'],
        default='auto',
        help='Device to use for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='experiments',
        help='Directory to save results'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to save/load model checkpoint'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=100,
        help='Log interval in batches'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.1,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--compute_tsne',
        action='store_true',
        help='Compute t-SNE embeddings during testing'
    )
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> str:
    """
    Get appropriate device based on argument and availability.
    
    Args:
        device_arg: Device argument ('cuda', 'cpu', 'mps', 'auto').
    
    Returns:
        Device string to use.
    """
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_arg


def args_to_config(args: argparse.Namespace) -> Config:
    """
    Convert parsed arguments to Config dataclasses.
    
    Args:
        args: Parsed command-line arguments.
    
    Returns:
        Config object containing all configuration.
    """
    model_config = ModelConfig(
        input_dim=784,
        hidden_dims=args.hidden_dims,
        num_classes=10,
        activation=args.activation,
        dropout=args.dropout,
        use_bn=args.use_bn
    )
    
    training_config = TrainingConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        l1_lambda=args.l1_lambda,
        scheduler=args.scheduler,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_patience=args.scheduler_patience,
        early_stop_patience=args.early_stop_patience,
        checkpoint_path=args.checkpoint_path
    )
    
    device = get_device(args.device)
    
    experiment_config = ExperimentConfig(
        device=device,
        seed=args.seed,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        val_split=args.val_split,
        num_workers=args.num_workers
    )
    
    return Config(
        model=model_config,
        training=training_config,
        experiment=experiment_config
    )


def save_config(config: Config, save_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration object.
        save_path: Path to save the configuration.
    """
    config_dict = {
        'model': {
            'input_dim': config.model.input_dim,
            'hidden_dims': config.model.hidden_dims,
            'num_classes': config.model.num_classes,
            'activation': config.model.activation,
            'dropout': config.model.dropout,
            'use_bn': config.model.use_bn
        },
        'training': {
            'lr': config.training.lr,
            'epochs': config.training.epochs,
            'batch_size': config.training.batch_size,
            'optimizer': config.training.optimizer,
            'momentum': config.training.momentum,
            'weight_decay': config.training.weight_decay,
            'l1_lambda': config.training.l1_lambda,
            'scheduler': config.training.scheduler,
            'early_stop_patience': config.training.early_stop_patience
        },
        'experiment': {
            'device': config.experiment.device,
            'seed': config.experiment.seed,
            'val_split': config.experiment.val_split
        }
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved configuration to {save_path}")


def main() -> None:
    """
    Main function to run training and/or testing.
    """
    args = parse_arguments()
    
    config = args_to_config(args)
    
    set_seed(config.experiment.seed)
    
    print("=" * 70)
    print("MNIST MLP Classification")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Device: {config.experiment.device}")
    print(f"Random seed: {config.experiment.seed}")
    
    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size=config.training.batch_size,
        val_split=config.experiment.val_split,
        num_workers=config.experiment.num_workers,
        seed=config.experiment.seed
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Validation: {len(val_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")
    
    model = MLP(
        input_dim=config.model.input_dim,
        hidden_dims=config.model.hidden_dims,
        num_classes=config.model.num_classes,
        activation=config.model.activation,
        dropout=config.model.dropout,
        use_bn=config.model.use_bn
    )
    
    print(f"\n{model.get_layer_info()}")
    
    config_path = os.path.join(config.experiment.save_dir, 'config.json')
    save_config(config, config_path)
    
    if args.mode in ['train', 'both']:
        print("\n" + "=" * 70)
        print("TRAINING")
        print("=" * 70)
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.training,
            device=config.experiment.device,
            log_interval=config.experiment.log_interval
        )
        
        plot_path = os.path.join(config.experiment.save_dir, 'training_curves.png')
        plot_training_curves(history, save_path=plot_path, show=False)
        
        if 'learning_rates' in history and len(history['learning_rates']) > 0:
            lr_plot_path = os.path.join(config.experiment.save_dir, 'lr_schedule.png')
            plot_learning_rate_schedule(
                history['learning_rates'],
                save_path=lr_plot_path,
                show=False
            )
    
    if args.mode in ['test', 'both']:
        print("\n" + "=" * 70)
        print("TESTING")
        print("=" * 70)
        
        results = test_model(
            model=model,
            test_loader=test_loader,
            device=config.experiment.device,
            save_dir=config.experiment.save_dir,
            checkpoint_path=config.training.checkpoint_path,
            compute_tsne_emb=args.compute_tsne
        )
        
        cm_path = os.path.join(config.experiment.save_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            np.array(results['confusion_matrix']),
            save_path=cm_path,
            show=False
        )
        
        if 'tsne_embeddings' in results:
            tsne_path = os.path.join(config.experiment.save_dir, 'tsne.png')
            plot_tsne(
                np.array(results['tsne_embeddings']),
                np.array(results['tsne_labels']),
                save_path=tsne_path,
                show=False
            )
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
