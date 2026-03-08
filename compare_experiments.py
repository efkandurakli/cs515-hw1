"""
Compare and visualize results from multiple experiments.

This script loads results from all experiments and creates comparison plots
for easy analysis and reporting.

Usage:
    python compare_experiments.py
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import glob


def load_experiment_results(experiments_dir: str = 'experiments') -> Dict:
    """
    Load results from all experiment directories.
    
    Args:
        experiments_dir: Base directory containing experiment folders.
    
    Returns:
        Dictionary mapping experiment names to their results.
    """
    results = {}
    
    for exp_dir in glob.glob(os.path.join(experiments_dir, '*')):
        if not os.path.isdir(exp_dir):
            continue
        
        exp_name = os.path.basename(exp_dir)
        
        # Try to find history file in checkpoints directory
        history_path = os.path.join('checkpoints', f'{exp_name}_history.json')
        if not os.path.exists(history_path):
            # Fallback: look for history file in experiment directory
            history_files = glob.glob(os.path.join(exp_dir, '*_history.json'))
            if history_files:
                history_path = history_files[0]
            else:
                history_path = None
        
        test_results_path = os.path.join(exp_dir, 'test_results.json')
        config_path = os.path.join(exp_dir, 'config.json')
        
        exp_data = {}
        
        if history_path and os.path.exists(history_path):
            with open(history_path, 'r') as f:
                exp_data['history'] = json.load(f)
        
        if os.path.exists(test_results_path):
            with open(test_results_path, 'r') as f:
                exp_data['test_results'] = json.load(f)
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                exp_data['config'] = json.load(f)
        
        if exp_data:
            results[exp_name] = exp_data
    
    return results


def plot_comparison_curves(results: Dict, save_dir: str = 'experiments/comparisons') -> None:
    """
    Create comparison plots for different experiment groups.
    
    Args:
        results: Dictionary of experiment results.
        save_dir: Directory to save comparison plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    groups = {
        'architecture': ['arch_'],
        'activation': ['activation_'],
        'dropout': ['dropout_'],
        'batch_norm': ['_bn'],
        'optimizer': ['optimizer_'],
        'scheduler': ['scheduler_'],
        'regularization': ['reg_']
    }
    
    for group_name, patterns in groups.items():
        group_results = {}
        
        for exp_name, exp_data in results.items():
            if any(pattern in exp_name for pattern in patterns):
                if 'history' in exp_data:
                    group_results[exp_name] = exp_data['history']
        
        if not group_results:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for exp_name, history in group_results.items():
            if 'val_losses' in history and len(history['val_losses']) > 0:
                epochs = range(1, len(history['val_losses']) + 1)
                axes[0].plot(epochs, history['val_losses'], label=exp_name, linewidth=2)
        
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Validation Loss', fontsize=12)
        axes[0].set_title(f'{group_name.title()} Comparison - Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=9, loc='best')
        axes[0].grid(True, alpha=0.3)
        
        for exp_name, history in group_results.items():
            if 'val_accs' in history and len(history['val_accs']) > 0:
                epochs = range(1, len(history['val_accs']) + 1)
                axes[1].plot(epochs, history['val_accs'], label=exp_name, linewidth=2)
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Validation Accuracy (%)', fontsize=12)
        axes[1].set_title(f'{group_name.title()} Comparison - Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=9, loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{group_name}_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {group_name} comparison to {save_path}")
        plt.close()


def create_summary_table(results: Dict, save_path: str = 'experiments/summary_table.txt') -> None:
    """
    Create a summary table of all experiments.
    
    Args:
        results: Dictionary of experiment results.
        save_path: Path to save summary table.
    """
    lines = []
    lines.append("=" * 120)
    lines.append(f"{'Experiment Name':<40} {'Val Acc (%)':<12} {'Test Acc (%)':<12} {'Best Epoch':<12} {'Parameters':<12}")
    lines.append("=" * 120)
    
    for exp_name, exp_data in sorted(results.items()):
        val_acc = '-'
        test_acc = '-'
        best_epoch = '-'
        
        if 'history' in exp_data:
            history = exp_data['history']
            if 'best_val_acc' in history:
                val_acc = f"{history['best_val_acc']:.2f}"
            if 'best_epoch' in history:
                best_epoch = str(history['best_epoch'])
        
        if 'test_results' in exp_data:
            test_results = exp_data['test_results']
            if 'test_accuracy' in test_results:
                test_acc = f"{test_results['test_accuracy']:.2f}"
        
        params = '-'
        if 'config' in exp_data and 'model' in exp_data['config']:
            hidden_dims = exp_data['config']['model'].get('hidden_dims', [])
            params = str(hidden_dims)
        
        lines.append(f"{exp_name:<40} {val_acc:<12} {test_acc:<12} {best_epoch:<12} {params:<12}")
    
    lines.append("=" * 120)
    
    summary = '\n'.join(lines)
    print(summary)
    
    with open(save_path, 'w') as f:
        f.write(summary)
    
    print(f"\nSaved summary table to {save_path}")


def plot_best_vs_worst(results: Dict, save_dir: str = 'experiments/comparisons') -> None:
    """
    Plot comparison of best and worst performing models.
    
    Args:
        results: Dictionary of experiment results.
        save_dir: Directory to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    accuracies = []
    for exp_name, exp_data in results.items():
        if 'test_results' in exp_data and 'test_accuracy' in exp_data['test_results']:
            accuracies.append((exp_name, exp_data['test_results']['test_accuracy']))
    
    if len(accuracies) < 2:
        print("Not enough experiments with test results for best vs worst comparison")
        return
    
    accuracies.sort(key=lambda x: x[1])
    worst_name, worst_acc = accuracies[0]
    best_name, best_acc = accuracies[-1]
    
    print(f"\nBest model: {best_name} (Test Acc: {best_acc:.2f}%)")
    print(f"Worst model: {worst_name} (Test Acc: {worst_acc:.2f}%)")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, label, color in [(best_name, 'Best', 'green'), (worst_name, 'Worst', 'red')]:
        if name in results and 'history' in results[name]:
            history = results[name]['history']
            if 'val_losses' in history:
                epochs = range(1, len(history['val_losses']) + 1)
                axes[0].plot(epochs, history['val_losses'], label=f'{label}: {name}', 
                           linewidth=2, color=color)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Best vs Worst Model - Validation Loss', fontsize=14, fontweight='bold')
    if axes[0].get_lines():
        axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    for name, label, color in [(best_name, 'Best', 'green'), (worst_name, 'Worst', 'red')]:
        if name in results and 'history' in results[name]:
            history = results[name]['history']
            if 'val_accs' in history:
                epochs = range(1, len(history['val_accs']) + 1)
                axes[1].plot(epochs, history['val_accs'], label=f'{label}: {name}', 
                           linewidth=2, color=color)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[1].set_title('Best vs Worst Model - Validation Accuracy', fontsize=14, fontweight='bold')
    if axes[1].get_lines():
        axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'best_vs_worst.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved best vs worst comparison to {save_path}")
    plt.close()


def main():
    """Main function to compare all experiments."""
    print("=" * 70)
    print("MNIST MLP Experiment Comparison")
    print("=" * 70)
    
    print("\nLoading experiment results...")
    results = load_experiment_results('experiments')
    
    if not results:
        print("No experiment results found in 'experiments/' directory.")
        print("Run some experiments first using main.py or run_experiments.sh")
        return
    
    print(f"Found {len(results)} experiments\n")
    
    print("Creating summary table...")
    create_summary_table(results)
    
    print("\nCreating comparison plots...")
    plot_comparison_curves(results)
    
    print("\nCreating best vs worst comparison...")
    plot_best_vs_worst(results)
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - experiments/summary_table.txt")
    print("  - experiments/comparisons/*.png")


if __name__ == '__main__':
    main()
