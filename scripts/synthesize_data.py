#!/usr/bin/env python3
"""
Data synthesis script for audio mixing training.

This script processes MUSDB18HQ multitrack songs, chunks them, injects gain errors,
and generates training data with both flawed and reference mixes.
"""

import random
from pathlib import Path

import hydra
from omegaconf import DictConfig

import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.synthesis import process_split


@hydra.main(version_base=None, config_path="../configs/data", config_name="synthesis")
def main(cfg: DictConfig) -> None:
    """Main function to synthesize training data."""
    print("Starting data synthesis...")
    print(f"Configuration: {cfg}")

    # Set up random number generator
    rng = random.Random(cfg.rng.seed)

    # Process both splits
    for split in cfg.splits:
        print(f"\nProcessing {split} split...")
        process_split(split, cfg.paths.musdb_root, cfg.paths.output_root, cfg, rng)

    print("\nData synthesis complete!")


if __name__ == "__main__":
    main()
