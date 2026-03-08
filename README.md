# MNIST MLP Classification

A comprehensive implementation of Multi-Layer Perceptron (MLP) for MNIST digit classification with extensive hyperparameter analysis and experimentation capabilities.

## Project Overview

This project implements a flexible MLP architecture for classifying handwritten digits from the MNIST dataset. The implementation includes support for various activation functions, dropout, batch normalization, different optimizers, learning rate schedulers, and regularization techniques.

## Features

- **Flexible MLP Architecture**: Configurable number of layers and hidden dimensions
- **Multiple Activation Functions**: ReLU and GELU
- **Regularization Techniques**: Dropout, Batch Normalization, L1/L2 regularization
- **Training Framework**: 
  - Multiple optimizers (SGD, Adam, AdamW)
  - Learning rate schedulers (StepLR, ReduceLROnPlateau, CosineAnnealingLR)
  - Early stopping based on validation loss
  - Model checkpointing
- **Comprehensive Evaluation**: 
  - Accuracy, confusion matrix, classification report
  - t-SNE visualization of learned features
- **Clean Code Structure**: 
  - Type hints and docstrings throughout
  - Dataclasses for configuration
  - Modular design

## Project Structure

```
cs515/
├── main.py              # Entry point with argparse CLI
├── train.py             # Training loop implementation
├── test.py              # Testing and evaluation
├── parameters.py        # Configuration dataclasses
├── models/
│   ├── __init__.py
│   └── mlp.py          # MLP model implementation
├── utils/
│   ├── __init__.py
│   ├── data_loader.py  # MNIST data loading
│   └── visualization.py # Plotting utilities
├── experiments/         # Saved experiment results
├── checkpoints/         # Model checkpoints
├── requirements.txt     # Dependencies
├── README.md           # This file
└── REPORT.md           # Analysis report
```

## Installation

1. Clone or download this repository:
```bash
cd cs515
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The project requires:
- torch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- scikit-learn >= 1.3.0
- tqdm >= 4.65.0

## Usage

### Basic Training

Train a basic MLP with default settings:

```bash
python main.py --mode train
```

### Custom Architecture

Train with custom hidden layers:

```bash
python main.py --mode train --hidden_dims 512 256 128 --activation gelu --dropout 0.2 --use_bn
```

### Training with Different Optimizers

```bash
# Adam optimizer
python main.py --mode train --optimizer adam --lr 0.001

# SGD with momentum
python main.py --mode train --optimizer sgd --lr 0.01 --momentum 0.9

# AdamW with weight decay
python main.py --mode train --optimizer adamw --lr 0.001 --weight_decay 0.01
```

### Learning Rate Schedulers

```bash
# StepLR scheduler
python main.py --mode train --scheduler step --scheduler_step_size 10 --scheduler_gamma 0.1

# ReduceLROnPlateau
python main.py --mode train --scheduler plateau --scheduler_patience 5

# CosineAnnealingLR
python main.py --mode train --scheduler cosine
```

### Regularization

```bash
# L2 regularization (weight decay)
python main.py --mode train --weight_decay 0.0001

# L1 regularization
python main.py --mode train --l1_lambda 0.0001

# Dropout
python main.py --mode train --dropout 0.3
```

### Testing

Test a trained model:

```bash
python main.py --mode test --checkpoint_path checkpoints/best_model.pth
```

Test with t-SNE visualization:

```bash
python main.py --mode test --checkpoint_path checkpoints/best_model.pth --compute_tsne
```

### Train and Test

Run both training and testing:

```bash
python main.py --mode both --hidden_dims 256 128 --activation relu --epochs 30
```

## Command-Line Arguments

### Model Architecture
- `--hidden_dims`: Hidden layer dimensions (e.g., `256 128`) [default: 256 128]
- `--activation`: Activation function (`relu` or `gelu`) [default: relu]
- `--dropout`: Dropout probability [default: 0.0]
- `--use_bn`: Use batch normalization (flag)

### Training Configuration
- `--lr`: Learning rate [default: 0.001]
- `--epochs`: Number of training epochs [default: 50]
- `--batch_size`: Batch size [default: 128]
- `--optimizer`: Optimizer type (`sgd`, `adam`, `adamw`) [default: adam]
- `--momentum`: Momentum for SGD [default: 0.9]
- `--weight_decay`: L2 regularization coefficient [default: 0.0]
- `--l1_lambda`: L1 regularization coefficient [default: 0.0]
- `--scheduler`: LR scheduler (`step`, `plateau`, `cosine`) [default: None]
- `--early_stop_patience`: Early stopping patience [default: 10]

### Experiment Settings
- `--mode`: Mode (`train`, `test`, `both`) [default: train]
- `--device`: Device (`cuda`, `cpu`, `mps`, `auto`) [default: auto]
- `--seed`: Random seed [default: 42]
- `--save_dir`: Directory to save results [default: experiments]
- `--checkpoint_path`: Checkpoint path [default: checkpoints/best_model.pth]
- `--compute_tsne`: Compute t-SNE embeddings (flag)

## Example Experiments

### 1. Architecture Analysis

Test different network depths:

```bash
# 2 layers
python main.py --mode train --hidden_dims 256 128 --save_dir experiments/arch_2layers

# 3 layers
python main.py --mode train --hidden_dims 512 256 128 --save_dir experiments/arch_3layers

# 4 layers
python main.py --mode train --hidden_dims 512 256 128 64 --save_dir experiments/arch_4layers
```

### 2. Activation Function Comparison

```bash
# ReLU
python main.py --mode train --activation relu --save_dir experiments/act_relu

# GELU
python main.py --mode train --activation gelu --save_dir experiments/act_gelu
```

### 3. Dropout Analysis

```bash
# No dropout
python main.py --mode train --dropout 0.0 --save_dir experiments/dropout_0.0

# Dropout 0.3
python main.py --mode train --dropout 0.3 --save_dir experiments/dropout_0.3

# Dropout 0.5
python main.py --mode train --dropout 0.5 --save_dir experiments/dropout_0.5
```

### 4. Batch Normalization Study

```bash
# Without BN
python main.py --mode train --save_dir experiments/no_bn

# With BN
python main.py --mode train --use_bn --save_dir experiments/with_bn
```

## Output Files

After training and testing, the following files are generated:

- `checkpoints/best_model.pth`: Best model checkpoint
- `experiments/config.json`: Configuration used
- `experiments/training_curves.png`: Training/validation loss and accuracy plots
- `experiments/confusion_matrix.png`: Confusion matrix heatmap
- `experiments/test_results.json`: Test metrics
- `experiments/tsne.png`: t-SNE visualization (if enabled)
- `checkpoints/best_model_history.json`: Training history

## Code Structure

### Parameters (`parameters.py`)

Defines dataclasses for configuration:
- `ModelConfig`: Architecture parameters
- `TrainingConfig`: Training settings
- `ExperimentConfig`: Experiment settings

### Model (`models/mlp.py`)

Implements the MLP class with:
- Dynamic layer construction using `nn.ModuleList`
- Support for various activation functions
- Optional batch normalization and dropout

### Training (`train.py`)

Provides:
- Training loop with progress bars
- Early stopping mechanism
- Model checkpointing
- Support for various optimizers and schedulers
- L1/L2 regularization

### Testing (`test.py`)

Includes:
- Model evaluation on test set
- Confusion matrix computation
- Classification report generation
- Feature extraction for t-SNE

### Visualization (`utils/visualization.py`)

Plotting functions for:
- Training curves
- Confusion matrices
- t-SNE embeddings
- Comparison plots

## Research Questions Addressed

This implementation enables analysis of:

1. **Architecture Impact**: Number of layers and hidden dimensions
2. **Activation Functions**: ReLU vs GELU performance
3. **Dropout Effect**: Impact on overfitting and generalization
4. **Batch Normalization**: Effect on training stability and performance
5. **Training Framework**: Optimizer and scheduler comparison
6. **Regularization**: L1/L2 regularization effectiveness

## Results Interpretation

See `REPORT.md` for detailed analysis and findings from various experiments.

## Tips for Best Results

1. **Start Simple**: Begin with a baseline configuration and gradually add complexity
2. **Monitor Validation Loss**: Use early stopping to prevent overfitting
3. **Learning Rate**: Often the most important hyperparameter to tune
4. **Batch Normalization**: Usually helps with deeper networks
5. **Dropout**: More effective with larger models prone to overfitting
6. **Regularization**: Start with small values (e.g., 1e-4) and adjust

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python main.py --batch_size 64
```

### Slow Training

Reduce number of workers or use GPU:
```bash
python main.py --num_workers 2 --device cuda
```

### Poor Performance

Try increasing model capacity or adjusting learning rate:
```bash
python main.py --hidden_dims 512 256 128 --lr 0.0001
```

## References

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Deep Learning Tutorial: https://github.com/SU-Intelligent-systems-Lab/Deep-learning

## License

This project is for educational purposes as part of CS515 coursework.

## Author

Implemented for CS515 HW1a: MNIST Classification with MLP
