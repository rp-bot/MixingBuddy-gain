"""
Gating and quality filtering functions for data synthesis.

This module handles filtering of audio chunks based on RMS energy
and activity levels to ensure only high-quality training samples.
"""

import numpy as np
from typing import Dict


def compute_rms_dbfs(audio: np.ndarray) -> float:
    """Calculate RMS energy in dBFS for gating decisions.

    Args:
        audio: Audio array

    Returns:
        RMS energy in dBFS
    """
    if len(audio) == 0:
        return -np.inf

    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return -np.inf

    # Convert to dBFS (assuming full scale is 1.0)
    return 20 * np.log10(rms)


def apply_gating(chunk_stems: Dict[str, np.ndarray], config) -> bool:
    """Check if chunk meets minimum RMS thresholds and active frame ratio.

    Args:
        chunk_stems: Dict with stem arrays for the chunk
        config: Configuration object containing gating and audio parameters

    Returns:
        True if chunk passes gating, False otherwise
    """
    # Check mixture RMS
    mixture = sum(chunk_stems.values())
    mixture_rms = compute_rms_dbfs(mixture)
    if mixture_rms < config.gating.mixture_min_rms_dbfs:
        return False

    # Check individual stem RMS
    for stem_name, stem_audio in chunk_stems.items():
        stem_rms = compute_rms_dbfs(stem_audio)
        if stem_rms < config.gating.stem_min_rms_dbfs:
            return False

    # Check active frame ratio (using configurable frame duration)
    frame_samples = int(config.frame.ms / 1000.0 * config.audio.sample_rate)
    active_frames = 0
    total_frames = 0

    for stem_name, stem_audio in chunk_stems.items():
        for i in range(0, len(stem_audio) - frame_samples + 1, frame_samples):
            frame = stem_audio[i : i + frame_samples]
            frame_rms = compute_rms_dbfs(frame)
            total_frames += 1
            if frame_rms > config.gating.stem_min_rms_dbfs:
                active_frames += 1

    if total_frames == 0:
        return False

    active_ratio = active_frames / total_frames
    return active_ratio >= config.gating.min_active_frame_ratio
