"""
Audio processing functions for data synthesis.

This module handles loading and chunking of multitrack audio data,
specifically for MUSDB18HQ dataset processing.
"""

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Dict, Generator, Tuple, Union

from src.utils.audio_utils import to_mono


def load_track_stems(
    track_path: Union[str, Path], sample_rate: int
) -> Dict[str, np.ndarray]:
    """Load stem WAV files from a track directory.

    Args:
        track_path: Path to track directory containing stem files
        sample_rate: Target sample rate for loading

    Returns:
        Dict with stems: {vocals, drums, bass, other} as numpy arrays
    """
    track_path = Path(track_path)
    stems = {}

    for stem_name in ["vocals", "drums", "bass", "other"]:
        stem_file = track_path / f"{stem_name}.wav"
        if stem_file.exists():
            # Load full audio file
            audio, orig_sr = sf.read(str(stem_file))
            audio = to_mono(audio)

            # Resample if needed
            if orig_sr != sample_rate:
                audio = librosa.resample(
                    audio,
                    orig_sr=orig_sr,
                    target_sr=sample_rate,
                    res_type="kaiser_best",
                )

            stems[stem_name] = audio.astype(np.float32)
        else:
            print(f"Warning: {stem_name} stem is missing for track {track_path.name}")
            # Create silent audio if stem is missing
            stems[stem_name] = np.zeros(
                int(30 * sample_rate), dtype=np.float32
            )  # 30s default

    return stems


def chunk_audio(
    stems_dict: Dict[str, np.ndarray], chunk_duration_sec: float, sample_rate: int
) -> Generator[Tuple[int, Dict[str, np.ndarray]], None, None]:
    """Split multitrack audio into non-overlapping chunks.

    Args:
        stems_dict: Dict with stem arrays
        chunk_duration_sec: Duration of each chunk in seconds
        sample_rate: Sample rate

    Yields:
        Tuple of (chunk_idx, chunk_stems_dict)
    """
    chunk_samples = int(chunk_duration_sec * sample_rate)

    # Find the minimum length across all stems
    min_length = min(len(stem) for stem in stems_dict.values())

    chunk_idx = 0
    start_sample = 0

    while start_sample + chunk_samples <= min_length:
        chunk_stems = {}
        for stem_name, stem_audio in stems_dict.items():
            chunk_stems[stem_name] = stem_audio[
                start_sample : start_sample + chunk_samples
            ]

        yield chunk_idx, chunk_stems
        chunk_idx += 1
        start_sample += chunk_samples
