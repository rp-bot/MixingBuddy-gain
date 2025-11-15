#!/usr/bin/env python3
"""
Create dataset with template variations.

This script loads existing JSONL files and applies random template variations
to each sample, maintaining the same dataset size but introducing diversity
in the wording of instructions and responses.
"""

import json
import random
from pathlib import Path

import hydra
from omegaconf import DictConfig

import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.text_generation import create_instruction, create_response


@hydra.main(version_base=None, config_path="../configs/data", config_name="08_musdb_expanded_augmented_variations")
def main(cfg: DictConfig) -> None:
    """Main function to create variations dataset."""
    print("Starting variations dataset creation...")
    print(f"Configuration: {cfg}")

    # Set up random number generator
    rng = random.Random(cfg.rng.seed)

    # Process both splits
    for split in cfg.splits:
        print(f"\nProcessing {split} split...")
        create_variations_for_split(split, cfg, rng)

    print("\nVariations dataset creation complete!")


def create_variations_for_split(split_name: str, config: DictConfig, rng: random.Random) -> None:
    """Create variations for a split's JSONL file.

    For each sample in the input JSONL, randomly selects one instruction template
    and one response template variation, maintaining the same number of samples
    but with diverse wording.

    Args:
        split_name: Name of the split ('train' or 'test')
        config: Configuration object
        rng: Random number generator
    """
    # Construct input JSONL file path
    if split_name == "train":
        input_jsonl_path = Path(config.train_jsonl_path)
    else:
        input_jsonl_path = Path(config.test_jsonl_path)

    # Construct output JSONL file path
    if split_name == "train":
        output_filename = config.output.train_samples
    else:
        output_filename = config.output.test_samples

    output_jsonl_path = Path(config.paths.output_root) / split_name / output_filename

    if not input_jsonl_path.exists():
        print(f"JSONL file not found: {input_jsonl_path}")
        return

    print(f"Loading samples from {input_jsonl_path}")

    # Load all samples
    samples = []
    with open(input_jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    print(f"Found {len(samples)} original samples to process")

    # Create output directory if needed
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Create shuffled template indices to ensure uniform distribution
    # Shuffle the instruction template indices
    instruction_indices = list(range(len(config.instruction_templates)))
    rng.shuffle(instruction_indices)
    
    # Create shuffled response template indices for each error category
    response_indices_by_category = {}
    for category in config.response_templates.keys():
        indices = list(range(len(config.response_templates[category])))
        rng.shuffle(indices)
        response_indices_by_category[category] = indices
    
    # Track usage counters for round-robin distribution
    instruction_counter = 0
    response_counters = {category: 0 for category in config.response_templates.keys()}

    # Process each sample and apply variations with uniform distribution
    varied_samples = []

    for i, sample in enumerate(samples):
        if i % 1000 == 0:
            print(f"Processing sample {i + 1}/{len(samples)}...")

        # Extract metadata from sample
        meta = sample["meta"]
        anchor_stem = meta["anchor_stem"]
        stems_present = meta["stems_present"]
        error_category = meta["error_category"]
        target_stem = meta["target_stem"]

        # Get duration from config
        duration_sec = config.chunk.sec

        # Create a copy of the sample
        varied_sample = sample.copy()

        # Select instruction template using round-robin from shuffled list
        instruction_idx = instruction_indices[instruction_counter % len(instruction_indices)]
        instruction_template = config.instruction_templates[instruction_idx]
        instruction_counter += 1
        
        # Format instruction
        stems_str = ", ".join(stems_present)
        new_instruction = instruction_template.format(
            duration_sec=duration_sec, stems_present=stems_str, anchor_stem=anchor_stem
        )

        # Select response template using round-robin from shuffled list for this category
        response_counter = response_counters[error_category]
        response_idx = response_indices_by_category[error_category][response_counter % len(response_indices_by_category[error_category])]
        response_template = config.response_templates[error_category][response_idx]
        response_counters[error_category] += 1
        
        # Format response
        if error_category == "no_error":
            new_response = response_template
        else:
            min_db, max_db = config.error.ranges_db[error_category]
            min_gain_db = abs(min_db)
            max_gain_db = abs(max_db)
            new_response = response_template.format(
                target_stem=target_stem, min_gain_db=min_gain_db, max_gain_db=max_gain_db
            )

        # Update the sample with varied text
        varied_sample["instruction"] = new_instruction
        varied_sample["response"] = new_response

        varied_samples.append(varied_sample)

    # Write varied samples to output file
    print(f"Writing {len(varied_samples)} varied samples to {output_jsonl_path}")
    with open(output_jsonl_path, "w") as f:
        for sample in varied_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created {len(varied_samples)} varied samples from {len(samples)} original samples")
    print(f"Dataset size maintained: {len(varied_samples)} samples with varied wording")


if __name__ == "__main__":
    main()

