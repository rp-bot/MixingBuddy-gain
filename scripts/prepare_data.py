#!/usr/bin/env python3
"""
Data preparation script for automatic mixing dataset.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset import create_sample_data

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main data preparation function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting data preparation...")

    # Print configuration
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    # Create data directories
    data_dir = Path(config.paths.data_dir)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Check if raw data exists
        if not any(raw_dir.iterdir()):
            logger.info("No raw data found. Creating sample data...")
            create_sample_data(processed_dir, num_samples=1000)
        else:
            logger.info("Processing existing raw data...")
            process_raw_data(raw_dir, processed_dir, config)

        # Generate data statistics
        stats = generate_data_statistics(processed_dir)

        # Save statistics
        stats_file = processed_dir / "data_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Data preparation completed. Statistics saved to {stats_file}")
        logger.info(f"Dataset statistics: {stats}")

    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise


def process_raw_data(raw_dir: Path, processed_dir: Path, config: DictConfig):
    """Process raw data files."""
    # This is a template function - customize based on your data format

    # Look for common data formats
    data_files = []
    for ext in ["*.json", "*.jsonl", "*.csv", "*.txt"]:
        data_files.extend(raw_dir.glob(ext))

    if not data_files:
        logger.warning("No data files found in raw directory")
        return

    logger.info(f"Found {len(data_files)} data files")

    all_data = []

    for file_path in data_files:
        logger.info(f"Processing {file_path}")

        if file_path.suffix == ".jsonl":
            data = load_jsonl(file_path)
        elif file_path.suffix == ".json":
            data = load_json(file_path)
        elif file_path.suffix == ".csv":
            data = load_csv(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_path.suffix}")
            continue

        all_data.extend(data)

    # Split data into train/validation/test
    train_data, val_data, test_data = split_data(all_data)

    # Save processed data
    save_jsonl(processed_dir / "train.jsonl", train_data)
    save_jsonl(processed_dir / "validation.jsonl", val_data)
    save_jsonl(processed_dir / "test.jsonl", test_data)

    logger.info(f"Processed {len(all_data)} samples")
    logger.info(
        f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}"
    )


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_json(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Assume it's a dict with data in a key
        return data.get("data", list(data.values())[0] if data else [])
    else:
        raise ValueError(f"Unexpected data format in {file_path}")


def load_csv(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from CSV file."""
    import pandas as pd

    df = pd.read_csv(file_path)

    # Convert to list of dicts
    data = df.to_dict("records")

    # Rename columns to standard format if needed
    for item in data:
        if "question" in item and "answer" in item:
            item["instruction"] = item.pop("question")
            item["response"] = item.pop("answer")
        elif "input" in item and "output" in item:
            # Already in correct format
            pass
        else:
            # Use first two columns as instruction/response
            cols = list(item.keys())
            if len(cols) >= 2:
                item["instruction"] = item[cols[0]]
                item["response"] = item[cols[1]]

    return data


def split_data(
    data: List[Dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1
) -> tuple:
    """Split data into train/validation/test sets."""
    import random

    # Shuffle data
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def save_jsonl(file_path: Path, data: List[Dict[str, Any]]):
    """Save data to JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def generate_data_statistics(processed_dir: Path) -> Dict[str, Any]:
    """Generate statistics about the processed dataset."""
    stats = {}

    for split in ["train", "validation", "test"]:
        file_path = processed_dir / f"{split}.jsonl"

        if not file_path.exists():
            continue

        data = load_jsonl(file_path)

        # Calculate statistics
        split_stats = {
            "num_samples": len(data),
            "avg_instruction_length": 0,
            "avg_response_length": 0,
            "total_tokens": 0,
        }

        if data:
            instruction_lengths = [
                len(item.get("instruction", "").split()) for item in data
            ]
            response_lengths = [len(item.get("response", "").split()) for item in data]

            split_stats.update(
                {
                    "avg_instruction_length": sum(instruction_lengths)
                    / len(instruction_lengths),
                    "avg_response_length": sum(response_lengths)
                    / len(response_lengths),
                    "total_tokens": sum(instruction_lengths) + sum(response_lengths),
                }
            )

        stats[split] = split_stats

    return stats


if __name__ == "__main__":
    main()
