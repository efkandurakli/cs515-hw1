"""
Parameter dataclasses for MNIST MLP training.

This module defines dataclasses for organizing configuration parameters
including model architecture, training settings, and experiment configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class ModelConfig:
    """
    Configuration for MLP model architecture.
    
    Attributes:
        input_dim: Input dimension (784 for flattened MNIST images).
        hidden_dims: List of hidden layer dimensions.
        num_classes: Number of output classes (10 for MNIST).
        activation: Activation function type ('relu' or 'gelu').
        dropout: Dropout probability (0.0 to 1.0).
        use_bn: Whether to use batch normalization before activation.
    """
    input_dim: int = 784
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    num_classes: int = 10
    activation: Literal['relu', 'gelu'] = 'relu'
    dropout: float = 0.0
    use_bn: bool = False


@dataclass
class TrainingConfig:
    """
    Configuration for training procedure.
    
    Attributes:
        lr: Learning rate.
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training and validation.
        optimizer: Optimizer type ('sgd', 'adam', 'adamw').
        momentum: Momentum for SGD optimizer.
        weight_decay: L2 regularization coefficient.
        l1_lambda: L1 regularization coefficient.
        scheduler: Learning rate scheduler type (None, 'step', 'plateau', 'cosine').
        scheduler_step_size: Step size for StepLR scheduler.
        scheduler_gamma: Multiplicative factor for StepLR scheduler.
        scheduler_patience: Patience for ReduceLROnPlateau scheduler.
        early_stop_patience: Number of epochs to wait before early stopping.
        checkpoint_path: Path to save best model checkpoint.
    """
    lr: float = 0.001
    epochs: int = 50
    batch_size: int = 128
    optimizer: Literal['sgd', 'adam', 'adamw'] = 'adam'
    momentum: float = 0.9
    weight_decay: float = 0.0
    l1_lambda: float = 0.0
    scheduler: Optional[Literal['step', 'plateau', 'cosine']] = None
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    scheduler_patience: int = 5
    early_stop_patience: int = 10
    checkpoint_path: str = 'checkpoints/best_model.pth'


@dataclass
class ExperimentConfig:
    """
    Configuration for experiment settings.
    
    Attributes:
        device: Device to use for training ('cuda' or 'cpu').
        seed: Random seed for reproducibility.
        save_dir: Directory to save experiment results.
        log_interval: Log training status every N batches.
        val_split: Validation split ratio from training data.
        num_workers: Number of workers for data loading.
    """
    device: str = 'cpu'
    seed: int = 42
    save_dir: str = 'experiments'
    log_interval: int = 100
    val_split: float = 0.1
    num_workers: int = 4


@dataclass
class Config:
    """
    Main configuration class combining all config types.
    
    Attributes:
        model: Model architecture configuration.
        training: Training procedure configuration.
        experiment: Experiment settings configuration.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
