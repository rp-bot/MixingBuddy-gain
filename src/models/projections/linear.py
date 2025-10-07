"""
Linear projection layer for mapping audio features to LLM embedding space.
"""

import torch
import torch.nn as nn
from typing import Optional


class LinearProjection(nn.Module):
    """Simple linear projection layer for audio-to-text feature mapping.

    This layer projects audio features from the encoder output space
    to the LLM embedding space for multimodal fusion.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        activation: Optional[str] = None,
    ):
        """Initialize the linear projection layer.

        Args:
            input_dim: Input feature dimension (e.g., 8*75=600 for Encodec)
            output_dim: Output embedding dimension (e.g., 3584 for Qwen2-7B)
            dropout: Dropout rate for regularization
            activation: Optional activation function ('relu', 'gelu', 'tanh', None)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Main projection layer
        self.projection = nn.Linear(input_dim, output_dim)

        # Optional dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

        # Optional activation
        if activation is not None:
            if activation.lower() == "relu":
                self.activation = nn.ReLU()
            elif activation.lower() == "gelu":
                self.activation = nn.GELU()
            elif activation.lower() == "tanh":
                self.activation = nn.Tanh()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.activation = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        # Use Xavier/Glorot initialization for better training stability
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the projection layer.

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

        # Apply projection
        x = self.projection(x)

        # Apply activation if specified
        if self.activation is not None:
            x = self.activation(x)

        # Apply dropout if specified
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)

        return x

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
            "dropout": self.dropout,
            "activation": str(self.activation) if self.activation else None,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


def create_linear_projection(
    input_dim: int,
    output_dim: int,
    dropout: float = 0.1,
    activation: Optional[str] = None,
) -> LinearProjection:
    """Factory function to create a linear projection layer.

    Args:
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
        dropout: Dropout rate
        activation: Optional activation function

    Returns:
        Initialized LinearProjection instance
    """
    return LinearProjection(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout=dropout,
        activation=activation,
    )
