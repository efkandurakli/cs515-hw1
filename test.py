"""
Testing and evaluation module for MNIST MLP classification.

This module implements model evaluation, metrics computation,
and feature visualization using t-SNE.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from tqdm import tqdm
import json
import os


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained PyTorch model.
        test_loader: Test data loader.
        device: Device to evaluate on ('cuda' or 'cpu').
    
    Returns:
        Tuple of (test_loss, test_accuracy, predictions, true_labels).
    """
    model.eval()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 10
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.
    
    Returns:
        Confusion matrix as numpy array.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    return cm


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None
) -> str:
    """
    Generate classification report with precision, recall, and F1-score.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        target_names: Optional list of class names.
    
    Returns:
        Classification report as string.
    """
    if target_names is None:
        target_names = [str(i) for i in range(10)]
    
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4
    )
    return report


def extract_features(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cpu',
    max_samples: int = 5000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from the penultimate layer of the model.
    
    This is useful for t-SNE visualization of learned representations.
    
    Args:
        model: Trained PyTorch model.
        data_loader: Data loader.
        device: Device to use ('cuda' or 'cpu').
        max_samples: Maximum number of samples to extract.
    
    Returns:
        Tuple of (features, labels) as numpy arrays.
    """
    model.eval()
    model = model.to(device)
    
    features_list = []
    labels_list = []
    
    def hook_fn(module, input, output):
        features_list.append(output.detach().cpu().numpy())
    
    handle = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'output_layer' not in name:
            last_linear = module
    
    if 'last_linear' in locals():
        handle = last_linear.register_forward_hook(hook_fn)
    
    sample_count = 0
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc='Extracting features'):
            if sample_count >= max_samples:
                break
            
            data = data.to(device)
            _ = model(data)
            
            labels_list.append(target.numpy())
            sample_count += data.size(0)
    
    if handle is not None:
        handle.remove()
    
    if len(features_list) > 0:
        features = np.concatenate(features_list, axis=0)[:max_samples]
        labels = np.concatenate(labels_list, axis=0)[:max_samples]
    else:
        features = np.array([])
        labels = np.array([])
    
    return features, labels


def compute_tsne(
    features: np.ndarray,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute t-SNE embedding of features.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features).
        perplexity: Perplexity parameter for t-SNE.
        n_iter: Number of iterations for t-SNE.
        random_state: Random seed.
    
    Returns:
        2D t-SNE embedding of shape (n_samples, 2).
    """
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        verbose=1
    )
    
    embeddings = tsne.fit_transform(features)
    return embeddings


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu',
    save_dir: str = 'experiments',
    checkpoint_path: Optional[str] = None,
    compute_tsne_emb: bool = False
) -> Dict:
    """
    Complete testing pipeline for model evaluation.
    
    Args:
        model: PyTorch model to test.
        test_loader: Test data loader.
        device: Device to test on ('cuda' or 'cpu').
        save_dir: Directory to save results.
        checkpoint_path: Path to model checkpoint to load.
        compute_tsne_emb: Whether to compute t-SNE embeddings.
    
    Returns:
        Dictionary containing test results including loss, accuracy,
        confusion matrix, and optionally t-SNE embeddings.
    """
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    test_loss, test_acc, predictions, true_labels = evaluate_model(
        model, test_loader, device
    )
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    cm = compute_confusion_matrix(true_labels, predictions)
    
    report = get_classification_report(true_labels, predictions)
    print(f"\nClassification Report:")
    print(report)
    
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    if compute_tsne_emb:
        print("\nComputing t-SNE embeddings...")
        features, labels = extract_features(model, test_loader, device)
        if len(features) > 0:
            embeddings = compute_tsne(features)
            results['tsne_embeddings'] = embeddings.tolist()
            results['tsne_labels'] = labels.tolist()
            print(f"Computed t-SNE for {len(labels)} samples")
    
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, 'test_results.json')
    
    with open(results_path, 'w') as f:
        results_serializable = results.copy()
        if 'classification_report' in results_serializable:
            results_serializable['classification_report'] = str(results['classification_report'])
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nSaved test results to {results_path}")
    
    return results
