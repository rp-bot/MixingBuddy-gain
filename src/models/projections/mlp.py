"""
Multi-layer perceptron projection layer for mapping audio features to LLM embedding space.

This module provides a more sophisticated projection layer with multiple hidden layers,
non-linear activations, and regularization techniques to increase information capacity.
"""

import torch
import torch.nn as nn
from typing import  List


class MLPProjection(nn.Module):
    """Multi-layer perceptron projection layer for audio-to-text feature mapping.

    This layer provides increased capacity compared to a simple linear projection
    through multiple hidden layers, non-linear activations, and regularization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [512, 1024],
        activation: str = "relu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_residual: bool = False,
    ):
        """Initialize the MLP projection layer.

        Args:
            input_dim: Input feature dimension (e.g., 128 for Encodec 32kHz)
            output_dim: Output embedding dimension (e.g., 3584 for Qwen2-7B)
            hidden_dims: List of hidden layer dimensions (e.g., [512, 1024])
            activation: Activation function ('relu', 'gelu', 'tanh')
            dropout: Dropout rate for regularization
            use_layer_norm: Whether to use LayerNorm after each hidden layer
            use_residual: Whether to use residual connections where dimensions match
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        # Build the network layers
        layers = []
        current_dim = input_dim

        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(current_dim, hidden_dim))

            # Activation
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "gelu":
                layers.append(nn.GELU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            # Layer normalization
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        # Create the network
        self.network = nn.Sequential(*layers)

        # Store layer dimensions for residual connections
        self.layer_dims = [input_dim] + hidden_dims + [output_dim]

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier/Glorot initialization for better training stability
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP projection layer.

        Args:
            x: Input tensor of shape (batch, time, input_dim) or (batch, time, channels, features)

        Returns:
            Projected tensor of shape (batch, time, output_dim)
        """
        # Handle different input shapes
        if x.dim() == 4:
            # Shape: (batch, time, channels, features) - flatten channels and features
            batch_size, time_steps, channels, features = x.shape
            x = x.reshape(
                batch_size, time_steps, -1
            )  # (batch, time, channels*features)
        elif x.dim() == 3:
            # Shape: (batch, time, features) - already correct
            pass
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D: {x.shape}")

        # Apply the network
        output = self.network(x)

        return output

    def get_output_dim(self) -> int:
        """Get the output dimension."""
        return self.output_dim

    def get_input_dim(self) -> int:
        """Get the input dimension."""
        return self.input_dim

    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "use_layer_norm": self.use_layer_norm,
            "use_residual": self.use_residual,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


def create_mlp_projection(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int] = [512, 1024],
    activation: str = "relu",
    dropout: float = 0.1,
    use_layer_norm: bool = True,
    use_residual: bool = False,
) -> MLPProjection:
    """Factory function to create an MLP projection layer.

    Args:
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function
        dropout: Dropout rate
        use_layer_norm: Whether to use LayerNorm
        use_residual: Whether to use residual connections

    Returns:
        Initialized MLPProjection instance
    """
    return MLPProjection(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
        use_residual=use_residual,
    )
