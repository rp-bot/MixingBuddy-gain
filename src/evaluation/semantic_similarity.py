"""
Custom evaluation metrics for mixing advice generation.

Includes:
1. Semantic similarity between generated and ground truth responses
"""

from typing import List
import torch
from sentence_transformers import SentenceTransformer


class SemanticSimilarityMetric:
    """Compute semantic similarity using sentence transformers."""

    def __init__(self, model_name: str):
        """
        Args:
            model_name: Name of the sentence transformer model to use
        """
        print(f"Loading semantic similarity model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.eval()

    def compute(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
        )
        return similarity.item()

    def compute_batch(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """
        Compute similarities for multiple text pairs.

        Args:
            texts1: List of first texts
            texts2: List of second texts

        Returns:
            List of similarity scores
        """
        if len(texts1) != len(texts2):
            raise ValueError("texts1 and texts2 must have the same length")

        similarities = []
        for t1, t2 in zip(texts1, texts2):
            similarities.append(self.compute(t1, t2))

        return similarities
