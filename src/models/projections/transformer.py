"""
Transformer-based projection layers for mapping audio features to LLM embedding space.

This module provides Transformer and Perceiver-style projection layers that can
capture temporal dependencies in audio sequences, improving the model's ability
to understand mixing relationships and identify problematic stems.
"""

import torch
import torch.nn as nn
from typing import Optional


class TransformerProjection(nn.Module):
    """Transformer-based projection for audio-to-text feature mapping.

    Uses self-attention to capture temporal dependencies in audio sequences,
    enabling better understanding of mixing relationships and stem identification.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        feedforward_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        max_seq_len: int = 2048,
    ):
        """Initialize the Transformer projection layer.

        Args:
            input_dim: Input feature dimension (e.g., 128 for Encodec 32kHz)
            output_dim: Output embedding dimension (e.g., 3584 for Qwen2-7B)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            feedforward_dim: Feedforward dimension (defaults to 4 * input_dim)
            dropout: Dropout rate for regularization
            use_layer_norm: Whether to use LayerNorm
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.max_seq_len = max_seq_len

        # Set feedforward dimension
        if feedforward_dim is None:
            feedforward_dim = 4 * input_dim
        self.feedforward_dim = feedforward_dim

        # Input projection to transformer dimension
        self.input_projection = nn.Linear(input_dim, input_dim)

        # Positional encoding (learnable embeddings)
        self.pos_embedding = nn.Embedding(max_seq_len, input_dim)

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=input_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection to LLM embedding space
        self.output_projection = nn.Linear(input_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer projection layer.

        Args:
            x: Input tensor of shape (batch, time, input_dim)

        Returns:
            Projected tensor of shape (batch, time, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        pos_embeds = self.pos_embedding(positions)
        x = x + pos_embeds

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Output projection
        output = self.output_projection(x)

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
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "dropout": self.dropout,
            "use_layer_norm": self.use_layer_norm,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feedforward network."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.norm1 = None
            self.norm2 = None

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer layer."""
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.dropout_layer(attn_output)

        if self.norm1 is not None:
            x = self.norm1(x)

        # Feedforward with residual connection
        ff_output = self.feedforward(x)
        x = x + ff_output

        if self.norm2 is not None:
            x = self.norm2(x)

        return x


class PerceiverResampler(nn.Module):
    """Perceiver-style resampler for efficient processing of long audio sequences.

    Uses learnable query tokens to downsample long sequences while preserving
    important information through cross-attention.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_queries: int = 64,
        num_layers: int = 2,
        num_heads: int = 8,
        feedforward_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """Initialize the Perceiver resampler.

        Args:
            input_dim: Input feature dimension
            output_dim: Output embedding dimension
            num_queries: Number of learnable query tokens
            num_layers: Number of cross-attention layers
            num_heads: Number of attention heads
            feedforward_dim: Feedforward dimension
            dropout: Dropout rate
            use_layer_norm: Whether to use LayerNorm
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        # Set feedforward dimension
        if feedforward_dim is None:
            feedforward_dim = 4 * input_dim
        self.feedforward_dim = feedforward_dim

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_queries, input_dim))

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList(
            [
                CrossAttentionLayer(
                    d_model=input_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_projection = nn.Linear(input_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        # Initialize query tokens
        nn.init.normal_(self.query_tokens, mean=0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Perceiver resampler.

        Args:
            x: Input tensor of shape (batch, time, input_dim)

        Returns:
            Projected tensor of shape (batch, num_queries, output_dim)
        """
        batch_size = x.shape[0]

        # Expand query tokens to batch size
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply cross-attention layers
        for layer in self.cross_attention_layers:
            queries = layer(queries, x)

        # Output projection
        output = self.output_projection(queries)

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
            "num_queries": self.num_queries,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "dropout": self.dropout,
            "use_layer_norm": self.use_layer_norm,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for Perceiver resampler."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        # Cross-attention: queries attend to input
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.norm1 = None
            self.norm2 = None

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the cross-attention layer.

        Args:
            queries: Query tokens (batch, num_queries, d_model)
            x: Input features (batch, time, d_model)

        Returns:
            Updated queries (batch, num_queries, d_model)
        """
        # Cross-attention: queries attend to input
        attn_output, _ = self.cross_attention(queries, x, x)
        queries = queries + self.dropout_layer(attn_output)

        if self.norm1 is not None:
            queries = self.norm1(queries)

        # Feedforward with residual connection
        ff_output = self.feedforward(queries)
        queries = queries + ff_output

        if self.norm2 is not None:
            queries = self.norm2(queries)

        return queries


def create_transformer_projection(
    input_dim: int,
    output_dim: int,
    num_layers: int = 3,
    num_heads: int = 8,
    feedforward_dim: Optional[int] = None,
    dropout: float = 0.1,
    use_layer_norm: bool = True,
    max_seq_len: int = 2048,
) -> TransformerProjection:
    """Factory function to create a Transformer projection layer.

    Args:
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        feedforward_dim: Feedforward dimension
        dropout: Dropout rate
        use_layer_norm: Whether to use LayerNorm
        max_seq_len: Maximum sequence length

    Returns:
        Initialized TransformerProjection instance
    """
    return TransformerProjection(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
        max_seq_len=max_seq_len,
    )


def create_perceiver_resampler(
    input_dim: int,
    output_dim: int,
    num_queries: int = 64,
    num_layers: int = 2,
    num_heads: int = 8,
    feedforward_dim: Optional[int] = None,
    dropout: float = 0.1,
    use_layer_norm: bool = True,
) -> PerceiverResampler:
    """Factory function to create a Perceiver resampler.

    Args:
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
        num_queries: Number of query tokens
        num_layers: Number of cross-attention layers
        num_heads: Number of attention heads
        feedforward_dim: Feedforward dimension
        dropout: Dropout rate
        use_layer_norm: Whether to use LayerNorm

    Returns:
        Initialized PerceiverResampler instance
    """
    return PerceiverResampler(
        input_dim=input_dim,
        output_dim=output_dim,
        num_queries=num_queries,
        num_layers=num_layers,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
    )
