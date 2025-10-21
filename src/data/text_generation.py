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
    error_ranges_db: Dict[str, List[float]],
    rng: random.Random,
) -> str:
    """Create response text from templates using dB ranges.

    Args:
        templates: Dict mapping error categories to response templates
        error_category: Type of error
        target_stem: Name of the target stem
        error_ranges_db: Dict mapping error categories to [min_db, max_db] ranges
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

    # Extract range values and convert to positive values for response
    min_db, max_db = error_ranges_db[error_category]
    min_gain_db = abs(min_db)
    max_gain_db = abs(max_db)

    # Format template with range parameters
    return template.format(
        target_stem=target_stem, min_gain_db=min_gain_db, max_gain_db=max_gain_db
    )
