"""
Text generation functions for data synthesis.

This module handles the creation of instruction and response text
from templates for training data.
"""

import random
from typing import Dict, List, Optional

QUIET_CATEGORIES = {"quiet", "very_quiet"}
LOUD_CATEGORIES = {"loud", "very_loud"}


def create_instruction(
    templates: List[str],
    duration_sec: float,
    stems_present: List[str],
    anchor_stem: str,
    target_stem: str,
    error_category: str,
    error_ranges_db: Dict[str, List[float]],
    rng: random.Random,
    template_override: Optional[str] = None,
) -> str:
    """Create instruction text from templates.

    Args:
        templates: List of instruction templates
        duration_sec: Duration of the audio segment
        stems_present: List of available stems
        anchor_stem: Name of the anchor stem
        target_stem: Name of the target stem
        error_category: Type of mixing error for this sample
        error_ranges_db: Mapping of error categories to dB ranges
        rng: Random number generator
        template_override: Optional template string to force usage

    Returns:
        Formatted instruction string
    """
    template = template_override or rng.choice(templates)
    stems_str = ", ".join(stems_present)
    target_requirement = build_target_requirement(
        target_stem=target_stem,
        error_category=error_category,
        error_ranges_db=error_ranges_db,
    )

    return template.format(
        duration_sec=duration_sec,
        stems_present=stems_str,
        anchor_stem=anchor_stem,
        target_stem=target_stem,
        target_requirement=target_requirement,
    )


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


def build_target_requirement(
    target_stem: str,
    error_category: str,
    error_ranges_db: Dict[str, List[float]],
) -> str:
    """Return instruction clause describing perceived issue and gain limits."""
    if error_category in QUIET_CATEGORIES | LOUD_CATEGORIES:
        range_vals = error_ranges_db.get(error_category, [])
        if len(range_vals) >= 2:
            abs_vals = sorted(abs(v) for v in range_vals)
            min_db, max_db = abs_vals[0], abs_vals[1]
        else:
            # Sensible fallback if config is missing values
            if error_category in QUIET_CATEGORIES:
                min_db, max_db = (3, 6) if error_category == "quiet" else (6, 12)
            else:
                min_db, max_db = (3, 6) if error_category == "loud" else (6, 12)

        if error_category in QUIET_CATEGORIES:
            return (
                f"The {target_stem} sounds too quiet; explain how much to boost it "
                f"(between +{min_db:g} and +{max_db:g} dB)."
            )
        else:
            return (
                f"The {target_stem} sounds too loud; explain how much to reduce it "
                f"(between -{min_db:g} and -{max_db:g} dB)."
            )

    # Default guidance for no-error or unknown categories
    return (
        f"The {target_stem} already feels balanced; confirm no adjustment is needed."
    )
