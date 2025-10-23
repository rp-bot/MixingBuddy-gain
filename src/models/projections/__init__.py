"""Projection layers for mapping audio features to LLM embedding space."""

from .linear import LinearProjection
from .mlp import MLPProjection

__all__ = ["LinearProjection", "MLPProjection"]
