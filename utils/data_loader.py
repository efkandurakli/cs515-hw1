"""
Data loading utilities for MNIST dataset.

This module provides functions to load and preprocess MNIST data,
including train/validation/test splits and data augmentation.
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from typing import Tuple, Optional


def get_mnist_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load MNIST dataset and create train/validation/test data loaders.
    
    The training set is split into training and validation sets according
    to val_split ratio. Standard normalization is applied with MNIST
    mean=0.1307 and std=0.3081.
    
    Args:
        data_dir: Directory to download/load MNIST data.
        batch_size: Batch size for data loaders.
        val_split: Fraction of training data to use for validation.
        num_workers: Number of subprocesses for data loading.
        seed: Random seed for reproducible train/val split.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    
    Example:
        >>> train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=64)
        >>> for images, labels in train_loader:
        ...     # Training loop
        ...     pass
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_mnist_datasets(
    data_dir: str = './data',
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load MNIST dataset and create train/validation/test datasets.
    
    This function is useful when you need direct access to datasets
    rather than data loaders.
    
    Args:
        data_dir: Directory to download/load MNIST data.
        val_split: Fraction of training data to use for validation.
        seed: Random seed for reproducible train/val split.
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    return train_subset, val_subset, test_dataset
