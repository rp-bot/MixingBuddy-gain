"""
Q-Former projection layer for mapping audio features to LLM embedding space.

Uses learnable query tokens with cross-attention to audio features, producing
a fixed-length output regardless of input audio length. Inspired by BLIP-2.
"""

import torch
import torch.nn as nn
from typing import Optional


class QFormerLayer(nn.Module):
    """Single Q-Former layer with self-attention and cross-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        feedforward_dim: Optional[int] = None,
    ):
        super().__init__()
        
        if feedforward_dim is None:
            feedforward_dim = hidden_dim * 4
        
        # Self-attention: queries attend to each other
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention: queries attend to audio features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        queries: torch.Tensor,
        audio_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: [batch, num_queries, hidden_dim]
            audio_features: [batch, time_steps, hidden_dim]
            audio_mask: Optional attention mask for audio features
            
        Returns:
            Updated queries: [batch, num_queries, hidden_dim]
        """
        # Self-attention with residual
        self_attn_out, _ = self.self_attn(
            query=queries,
            key=queries,
            value=queries,
        )
        queries = self.self_attn_norm(queries + self_attn_out)
        
        # Cross-attention with residual
        cross_attn_out, _ = self.cross_attn(
            query=queries,
            key=audio_features,
            value=audio_features,
            key_padding_mask=audio_mask,
        )
        queries = self.cross_attn_norm(queries + cross_attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(queries)
        queries = self.ffn_norm(queries + ffn_out)
        
        return queries


class QFormerProjection(nn.Module):
    """Q-Former based projection layer for audio-to-text feature mapping.
    
    Uses learnable query tokens that attend to audio features via cross-attention,
    producing a fixed-length output suitable for LLM input. This solves the
    variable-length problem between audio and text.
    
    Architecture:
        1. Project audio features to hidden dimension
        2. Learnable queries attend to audio via cross-attention
        3. Queries attend to each other via self-attention
        4. Project to LLM embedding space
    """

    def __init__(
        self,
        audio_dim: int = 1024,
        output_dim: int = 3584,
        hidden_dim: int = 1024,
        num_queries: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        feedforward_dim: Optional[int] = None,
    ):
        """Initialize the Q-Former projection layer.
        
        Args:
            audio_dim: Input audio feature dimension (e.g., 1024 for MERT)
            output_dim: Output embedding dimension (e.g., 3584 for Qwen2-7B)
            hidden_dim: Hidden dimension for transformer layers
            num_queries: Number of learnable query tokens (output sequence length)
            num_layers: Number of Q-Former transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate for regularization
            feedforward_dim: Feed-forward network dimension (default: 4 * hidden_dim)
        """
        super().__init__()
        
        self.audio_dim = audio_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Project audio features to hidden dimension if needed
        if audio_dim != hidden_dim:
            self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        else:
            self.audio_projection = nn.Identity()
        
        # Learnable query embeddings
        self.query_embeddings = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim) * 0.02
        )
        
        # Q-Former transformer layers
        self.layers = nn.ModuleList([
            QFormerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                feedforward_dim=feedforward_dim,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection to LLM space
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Layer norm before output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the Q-Former projection layer.
        
        Args:
            audio_features: Audio features from encoder
                Shape: [batch, time_steps, audio_dim] or [batch, time_steps, channels, features]
            audio_mask: Optional boolean mask for audio features (True = masked/padded)
                Shape: [batch, time_steps]
        
        Returns:
            Projected tensor of shape [batch, num_queries, output_dim]
        """
        # Handle different input shapes
        if audio_features.dim() == 4:
            # Shape: (batch, time, channels, features) - flatten channels and features
            batch_size, time_steps, channels, features = audio_features.shape
            audio_features = audio_features.reshape(batch_size, time_steps, -1)
        elif audio_features.dim() == 3:
            batch_size = audio_features.shape[0]
        else:
            raise ValueError(
                f"Expected 3D or 4D input, got {audio_features.dim()}D: {audio_features.shape}"
            )
        
        # Project audio to hidden dimension
        audio_features = self.audio_projection(audio_features)
        
        # Expand learnable queries for batch
        queries = self.query_embeddings.expand(batch_size, -1, -1)
        
        # Pass through Q-Former layers
        for layer in self.layers:
            queries = layer(queries, audio_features, audio_mask)
        
        # Normalize and project to output dimension
        queries = self.output_norm(queries)
        output = self.output_projection(queries)
        
        return output
    
    def get_output_dim(self) -> int:
        """Get the output dimension."""
        return self.output_dim
    
    def get_input_dim(self) -> int:
        """Get the input dimension."""
        return self.audio_dim
    
    def get_num_queries(self) -> int:
        """Get the number of output query tokens."""
        return self.num_queries
    
    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "audio_dim": self.audio_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "num_queries": self.num_queries,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


def create_qformer_projection(
    audio_dim: int = 1024,
    output_dim: int = 3584,
    hidden_dim: int = 1024,
    num_queries: int = 32,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    feedforward_dim: Optional[int] = None,
) -> QFormerProjection:
    """Factory function to create a Q-Former projection layer.
    
    Args:
        audio_dim: Input audio feature dimension
        output_dim: Output embedding dimension
        hidden_dim: Hidden dimension for transformer layers
        num_queries: Number of learnable query tokens
        num_layers: Number of Q-Former layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        feedforward_dim: Feed-forward dimension
    
    Returns:
        Initialized QFormerProjection instance
    """
    return QFormerProjection(
        audio_dim=audio_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        feedforward_dim=feedforward_dim,
    )

