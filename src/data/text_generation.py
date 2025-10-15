"""
Text generation functions for data synthesis.

This module handles the creation of instruction and response text
from templates for training data.
"""

import random
from typing import Dict, List


def create_instruction(
    templates: List[str],
    duration_sec: float,
    stems_present: List[str],
    rng: random.Random,
) -> str:
    """Create instruction text from templates.

    Args:
        templates: List of instruction templates
        duration_sec: Duration of the audio segment
        stems_present: List of available stems
        rng: Random number generator

    Returns:
        Formatted instruction string
    """
    template = rng.choice(templates)
    stems_str = ", ".join(stems_present)

    return template.format(duration_sec=duration_sec, stems_present=stems_str)


def create_response(
    templates: Dict[str, List[str]],
    error_category: str,
    target_stem: str,
    gain_db: float,
    rng: random.Random,
) -> str:
    """Create response text from templates.

    Args:
        templates: Dict mapping error categories to response templates
        error_category: Type of error
        target_stem: Name of the target stem
        gain_db: Gain applied in dB
        rng: Random number generator

    Returns:
        Formatted response string
    """
    if error_category == "no_error":
        template = rng.choice(templates["no_error"])
        return template

    # Get templates for the error category
    category_templates = templates[error_category]
    template = rng.choice(category_templates)

    # Format template with parameters
    if error_category in ["loud", "very_loud"]:
        # Use absolute value for loud categories
        abs_gain_db = abs(gain_db)
        return template.format(target_stem=target_stem, abs_gain_db=abs_gain_db)
    else:
        # Use actual gain for quiet categories
        return template.format(target_stem=target_stem, intended_gain_db=gain_db)
