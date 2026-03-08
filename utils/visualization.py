"""
Visualization utilities for training results and model analysis.

This module provides functions to create various plots including
training curves, confusion matrices, and t-SNE visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict
import os


def plot_training_curves(
    history: Dict,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary containing training history with keys:
                 'train_losses', 'val_losses', 'train_accs', 'val_accs'.
        save_path: Path to save the plot. If None, plot is not saved.
        show: Whether to display the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    axes[0].plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    if 'best_epoch' in history:
        axes[0].axvline(x=history['best_epoch'], color='g', linestyle='--',
                       label=f"Best Epoch ({history['best_epoch']})", linewidth=1.5)
        axes[0].legend(fontsize=11)
    
    axes[1].plot(epochs, history['train_accs'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_accs'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    if 'best_epoch' in history:
        axes[1].axvline(x=history['best_epoch'], color='g', linestyle='--',
                       label=f"Best Epoch ({history['best_epoch']})", linewidth=1.5)
        axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    normalize: bool = False
) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix as numpy array.
        class_names: List of class names. If None, uses numeric labels.
        save_path: Path to save the plot. If None, plot is not saved.
        show: Whether to display the plot.
        normalize: Whether to normalize the confusion matrix.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = 't-SNE Visualization of Learned Features'
) -> None:
    """
    Plot t-SNE embeddings colored by class labels.
    
    Args:
        embeddings: 2D t-SNE embeddings of shape (n_samples, 2).
        labels: Class labels of shape (n_samples,).
        save_path: Path to save the plot. If None, plot is not saved.
        show: Whether to display the plot.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[color],
            label=f'Class {label}',
            alpha=0.6,
            s=20
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_rate_schedule(
    learning_rates: List[float],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot learning rate schedule over epochs.
    
    Args:
        learning_rates: List of learning rates per epoch.
        save_path: Path to save the plot. If None, plot is not saved.
        show: Whether to display the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = range(1, len(learning_rates) + 1)
    ax.plot(epochs, learning_rates, 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning rate schedule to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    results_dict: Dict[str, Dict],
    metric: str = 'val_accs',
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None
) -> None:
    """
    Plot comparison of multiple experiments.
    
    Args:
        results_dict: Dictionary mapping experiment names to their history.
        metric: Metric to plot ('val_accs', 'val_losses', etc.).
        save_path: Path to save the plot. If None, plot is not saved.
        show: Whether to display the plot.
        title: Plot title. If None, auto-generated based on metric.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, history in results_dict.items():
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], label=name, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    
    if metric.endswith('_accs'):
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        default_title = 'Validation Accuracy Comparison'
    elif metric.endswith('_losses'):
        ax.set_ylabel('Loss', fontsize=12)
        default_title = 'Validation Loss Comparison'
    else:
        ax.set_ylabel(metric, fontsize=12)
        default_title = f'{metric} Comparison'
    
    ax.set_title(title if title else default_title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
