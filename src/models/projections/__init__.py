"""Projection layers for mapping audio features to LLM embedding space."""

from .linear import LinearProjection
from .mlp import MLPProjection
from .cross_attention import CrossAttentionProjection

__all__ = ["LinearProjection", "MLPProjection", "CrossAttentionProjection"]
