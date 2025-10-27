#!/usr/bin/env python3
"""
Edit synthesis JSONL script for regenerating instruction and response text.

This script loads existing JSONL files from synthesis and regenerates
instruction and response text using new templates from a config file,
without reprocessing the audio files.
"""

import json
import random
from pathlib import Path

import hydra
from omegaconf import DictConfig

import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.text_generation import create_instruction, create_response


@hydra.main(version_base=None, config_path="../configs/data", config_name="04_instructions_no_anchor")
def main(cfg: DictConfig) -> None:
    """Main function to edit synthesis JSONL files."""
    print("Starting JSONL editing...")
    print(f"Configuration: {cfg}")

    # Set up random number generator
    rng = random.Random(cfg.rng.seed)

    # Process both splits
    for split in cfg.splits:
        print(f"\nProcessing {split} split...")
        edit_split_jsonl(split, cfg, rng)

    print("\nJSONL editing complete!")


def edit_split_jsonl(split_name: str, config: DictConfig, rng: random.Random) -> None:
    """Edit instruction and response text for a split's JSONL file.

    Creates a new file with a suffix to prevent overwriting the original.
    If no output_suffix is specified in config, uses 'edited' as default.

    Args:
        split_name: Name of the split ('train' or 'test')
        config: Configuration object
        rng: Random number generator
    """
    # Construct input JSONL file path
    input_jsonl_path = (
        Path(config.paths.output_root)
        / split_name
        / config.output[f"{split_name}_samples"]
    )

    # Construct output JSONL file path with mandatory suffix
    output_filename = config.output[f"{split_name}_samples"]

    # Always add suffix to prevent overwriting original files

    suffix = config.output.output_suffix

    # Add suffix before file extension
    name_parts = output_filename.rsplit(".", 1)
    if len(name_parts) == 2:
        output_filename = f"{name_parts[0]}_{suffix}.{name_parts[1]}"
    else:
        output_filename = f"{output_filename}_{suffix}"

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

    print(f"Found {len(samples)} samples to process")

    # Process each sample
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"Processing sample {i + 1}/{len(samples)}...")

        # Extract metadata from sample
        meta = sample["meta"]
        anchor_stem = meta["anchor_stem"]
        stems_present = meta["stems_present"]
        error_category = meta["error_category"]
        target_stem = meta["target_stem"]

        # Get duration from config
        duration_sec = config.chunk.sec

        # Regenerate instruction
        new_instruction = create_instruction(
            config.instruction_templates, duration_sec, stems_present, anchor_stem, rng
        )

        # Regenerate response
        new_response = create_response(
            config.response_templates,
            error_category,
            target_stem,
            config.error.ranges_db,
            rng,
        )

        # Update sample
        sample["instruction"] = new_instruction
        sample["response"] = new_response

    # Write updated samples to output file
    print(f"Writing updated samples to {output_jsonl_path}")
    with open(output_jsonl_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Updated {len(samples)} samples in {output_jsonl_path}")


if __name__ == "__main__":
    main()
