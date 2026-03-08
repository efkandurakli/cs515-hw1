"""
Multi-Layer Perceptron (MLP) model for MNIST classification.

This module implements a flexible MLP architecture with configurable
number of layers, activation functions, batch normalization, and dropout.
"""

import torch
import torch.nn as nn
from torch.nn.modules.flatten import Flatten
from typing import List, Literal


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for classification tasks.
    
    This model consists of:
    - Flatten layer to convert 2D images to 1D vectors
    - Multiple fully connected hidden layers with configurable dimensions
    - Optional batch normalization before activation
    - Configurable activation functions (ReLU or GELU)
    - Optional dropout for regularization
    - Final output layer for classification
    
    Attributes:
        input_dim: Dimension of input features.
        hidden_dims: List of dimensions for hidden layers.
        num_classes: Number of output classes.
        activation: Type of activation function ('relu' or 'gelu').
        dropout: Dropout probability.
        use_bn: Whether to use batch normalization.
        flatten: Flatten layer to reshape input.
        layers: ModuleList containing all network layers.
    
    Example:
        >>> model = MLP(input_dim=784, hidden_dims=[256, 128], num_classes=10,
        ...             activation='relu', dropout=0.2, use_bn=True)
        >>> x = torch.randn(32, 1, 28, 28)  # Batch of MNIST images
        >>> output = model(x)  # Shape: (32, 10)
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 10,
        activation: Literal['relu', 'gelu'] = 'relu',
        dropout: float = 0.0,
        use_bn: bool = False
    ) -> None:
        """
        Initialize MLP model.
        
        Args:
            input_dim: Dimension of input features (784 for MNIST).
            hidden_dims: List of hidden layer dimensions.
            num_classes: Number of output classes (10 for MNIST).
            activation: Activation function type ('relu' or 'gelu').
            dropout: Dropout probability (0.0 to 1.0).
            use_bn: Whether to use batch normalization before activation.
        """
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.activation = activation
        self.dropout = dropout
        self.use_bn = use_bn
        
        self.flatten = Flatten()
        
        activation_fn = nn.ReLU() if activation == 'relu' else nn.GELU()
        
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layer_block = []
            
            layer_block.append(nn.Linear(dims[i], dims[i + 1]))
            
            if use_bn:
                layer_block.append(nn.BatchNorm1d(dims[i + 1]))
            
            layer_block.append(
                nn.ReLU() if activation == 'relu' else nn.GELU()
            )
            
            if dropout > 0.0:
                layer_block.append(nn.Dropout(dropout))
            
            self.layers.append(nn.Sequential(*layer_block))
        
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               or (batch_size, input_dim).
        
        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        x = self.flatten(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_layer(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """
        Calculate total number of trainable parameters.
        
        Returns:
            Total number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_info(self) -> str:
        """
        Get string representation of model architecture.
        
        Returns:
            String describing the model architecture.
        """
        info = f"MLP Architecture:\n"
        info += f"  Input dimension: {self.input_dim}\n"
        info += f"  Hidden dimensions: {self.hidden_dims}\n"
        info += f"  Output dimension: {self.num_classes}\n"
        info += f"  Activation: {self.activation}\n"
        info += f"  Dropout: {self.dropout}\n"
        info += f"  Batch Normalization: {self.use_bn}\n"
        info += f"  Total parameters: {self.get_num_parameters():,}\n"
        return info
