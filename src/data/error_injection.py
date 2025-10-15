"""
Error injection functions for data synthesis.

This module handles the simulation of gain errors in audio stems
to create flawed training examples.
"""

import random
import numpy as np
from typing import Dict, Tuple

from src.utils.audio_utils import db_to_linear


def sample_error_category(priors: Dict[str, float], rng: random.Random) -> str:
    """Sample an error category based on configured priors.

    Args:
        priors: Dict mapping category names to probabilities
        rng: Random number generator

    Returns:
        Selected error category
    """
    categories = list(priors.keys())
    probabilities = list(priors.values())
    return rng.choices(categories, weights=probabilities)[0]


def apply_gain_error(
    audio: np.ndarray, error_category: str, error_config: Dict, rng: random.Random
) -> Tuple[np.ndarray, float]:
    """Apply gain error to audio based on error category.

    Args:
        audio: Input audio array
        error_category: Type of error to apply
        error_config: Error configuration parameters
        rng: Random number generator

    Returns:
        Tuple of (modified_audio, actual_gain_db)
    """
    if error_category == "no_error":
        return audio, 0.0

    # Sample gain from category range
    gain_range = error_config["ranges_db"][error_category]
    gain_db = rng.uniform(gain_range[0], gain_range[1])

    # Apply gain limits
    gain_db = max(
        error_config["gain_limits_db"]["min"],
        min(error_config["gain_limits_db"]["max"], gain_db),
    )

    # Apply gain
    gain_linear = db_to_linear(gain_db)
    modified_audio = audio * gain_linear

    return modified_audio, gain_db
