"""
Cross-attention projection layer for fusing anchor and mix audio features.

Uses cross-attention where mix attends to anchor to produce a representation
of the mix in relation to the anchor, then projects to LLM embedding space.
"""

import torch
import torch.nn as nn


class CrossAttentionProjection(nn.Module):
    """Cross-attention based projection for audio-to-text feature mapping.

    The mix features attend to anchor features to understand relationships,
    then the attended representation is projected to LLM embedding space.
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize the cross-attention projection layer.

        Args:
            feature_dim: Input feature dimension per segment (channels * features)
            output_dim: Output embedding dimension (LLM hidden size)
            num_heads: Number of attention heads
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Cross-attention: mix (query) attends to anchor (key, value)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm after attention
        self.norm = nn.LayerNorm(feature_dim)

        # Final projection to LLM space
        self.projection = nn.Linear(feature_dim, output_dim)

        # Optional dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        anchor_features: torch.Tensor,
        mix_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through cross-attention projection.

        Args:
            anchor_features: Anchor audio features (batch, time, channels, features)
            mix_features: Mix audio features (batch, time, channels, features)

        Returns:
            Projected tensor of shape (batch, time, output_dim)
        """
        # Flatten channel and feature dimensions
        if anchor_features.dim() == 4:
            batch_size, time_steps, channels, features = anchor_features.shape
            anchor_flat = anchor_features.reshape(batch_size, time_steps, -1)
            mix_flat = mix_features.reshape(batch_size, time_steps, -1)
        else:
            anchor_flat = anchor_features
            mix_flat = mix_features

        # Cross-attention: mix attends to anchor
        # query=mix, key=anchor, value=anchor
        attn_output, _ = self.cross_attention(
            query=mix_flat,
            key=anchor_flat,
            value=anchor_flat,
        )

        # Residual connection + layer norm
        output = self.norm(mix_flat + attn_output)

        # Project to LLM embedding space
        output = self.projection(output)

        # Apply dropout if specified
        if self.dropout_layer is not None:
            output = self.dropout_layer(output)

        return output

    def get_output_dim(self) -> int:
        """Get the output dimension."""
        return self.output_dim

    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "feature_dim": self.feature_dim,
            "output_dim": self.output_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


def create_cross_attention_projection(
    feature_dim: int,
    output_dim: int,
    num_heads: int = 8,
    dropout: float = 0.1,
) -> CrossAttentionProjection:
    """Factory function to create a cross-attention projection layer.

    Args:
        feature_dim: Input feature dimension per segment
        output_dim: Output embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate

    Returns:
        Initialized CrossAttentionProjection instance
    """
    return CrossAttentionProjection(
        feature_dim=feature_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        dropout=dropout,
    )
