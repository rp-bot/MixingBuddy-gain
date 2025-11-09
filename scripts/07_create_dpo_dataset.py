#!/usr/bin/env python3
"""
Create DPO (Direct Preference Optimization) dataset from existing mixing dataset.

For each audio sample with a correct response (chosen), generate all possible
alternative incorrect responses (rejected) based on other stem-error combinations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig
from tqdm import tqdm


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], file_path: Path):
    """Save list of dictionaries to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def generate_response(
    target_stem: str,
    error_category: str,
    templates: Dict[str, List[str]],
    error_ranges: Dict[str, List[float]],
) -> str:
    """
    Generate a response for a given stem-error combination.
    
    Args:
        target_stem: The stem to mention in the response (e.g., "vocals", "drums")
        error_category: The error type (e.g., "quiet", "loud", "no_error")
        templates: Response templates from config
        error_ranges: dB ranges for each error category
        
    Returns:
        Generated response string
    """
    # Get template for this error category
    template = templates[error_category][0]  # Take first template
    
    # Special case: no_error doesn't reference a specific stem
    if error_category == "no_error":
        return template
    
    # Get the dB range for this error category
    error_range = error_ranges[error_category]
    
    # Calculate correction range (opposite of applied error)
    if error_category in ["quiet", "very_quiet"]:
        # Error was negative gain, so correction is positive increase
        min_correction = abs(error_range[1])  # abs(-3) = 3
        max_correction = abs(error_range[0])  # abs(-6) = 6
    else:  # loud, very_loud
        # Error was positive gain, so correction is reduction
        min_correction = error_range[0]  # 3 or 6
        max_correction = error_range[1]  # 6 or 12
    
    # Fill in the template
    response = template.format(
        target_stem=target_stem,
        min_gain_db=int(min_correction),
        max_gain_db=int(max_correction),
    )
    
    return response


def generate_all_alternative_responses(
    chosen_stem: str,
    chosen_error: str,
    stems_present: List[str],
    templates: Dict[str, List[str]],
    error_ranges: Dict[str, List[float]],
    error_categories: List[str],
) -> List[Tuple[str, str, str]]:
    """
    Generate all possible alternative (rejected) responses for a sample.
    
    Args:
        chosen_stem: The correct target stem
        chosen_error: The correct error category
        stems_present: List of stems available in this sample
        templates: Response templates from config
        error_ranges: dB ranges for each error category
        error_categories: List of all error categories
        
    Returns:
        List of tuples: (rejected_response, rejected_stem, rejected_error)
    """
    alternatives = []
    
    for error_cat in error_categories:
        if error_cat == "no_error":
            # no_error doesn't have a target stem
            # Skip if this is the chosen response
            if chosen_error == "no_error":
                continue
            response = generate_response(None, error_cat, templates, error_ranges)
            alternatives.append((response, "none", error_cat))
        else:
            # Generate for each available stem
            for stem in stems_present:
                # Skip if this is the chosen response
                if stem == chosen_stem and error_cat == chosen_error:
                    continue
                    
                response = generate_response(stem, error_cat, templates, error_ranges)
                alternatives.append((response, stem, error_cat))
    
    return alternatives


def create_dpo_pairs(
    original_data: List[Dict],
    templates: Dict[str, List[str]],
    error_ranges: Dict[str, List[float]],
    error_categories: List[str],
) -> List[Dict]:
    """
    Create DPO preference pairs from original dataset.
    
    Args:
        original_data: List of original training samples
        templates: Response templates from config
        error_ranges: dB ranges for each error category
        error_categories: List of all error categories
        
    Returns:
        List of DPO pairs (one per chosen-rejected combination)
    """
    dpo_pairs = []
    
    for sample in tqdm(original_data, desc="Creating DPO pairs"):
        # Extract metadata
        global_uid = sample["global_uid"]
        instruction = sample["instruction"]
        chosen_response = sample["response"]
        audio_path = sample["flawed_mix_path"]
        
        meta = sample["meta"]
        chosen_stem = meta["target_stem"]
        chosen_error = meta["error_category"]
        stems_present = meta["stems_present"]
        
        # Generate all alternative (rejected) responses
        alternatives = generate_all_alternative_responses(
            chosen_stem=chosen_stem,
            chosen_error=chosen_error,
            stems_present=stems_present,
            templates=templates,
            error_ranges=error_ranges,
            error_categories=error_categories,
        )
        
        # Create a DPO pair for each alternative
        for rejected_response, rejected_stem, rejected_error in alternatives:
            dpo_pair = {
                "global_uid": global_uid,
                "audio_path": audio_path,
                "instruction": instruction,
                "chosen": chosen_response,
                "rejected": rejected_response,
                "meta": {
                    "chosen_target_stem": chosen_stem,
                    "chosen_error_category": chosen_error,
                    "rejected_target_stem": rejected_stem,
                    "rejected_error_category": rejected_error,
                    "stems_present": stems_present,
                    "track_name": meta.get("track_name", ""),
                    "chunk_idx": meta.get("chunk_idx", 0),
                },
            }
            dpo_pairs.append(dpo_pair)
    
    return dpo_pairs


def generate_statistics(
    original_count: int,
    dpo_pairs: List[Dict],
    split_name: str,
) -> str:
    """
    Generate statistics about the DPO dataset.
    
    Args:
        original_count: Number of samples in original dataset
        dpo_pairs: List of generated DPO pairs
        split_name: Name of the split (train/test)
        
    Returns:
        Statistics string
    """
    stats = []
    stats.append(f"\n{'='*60}")
    stats.append(f"DPO Dataset Statistics - {split_name.upper()}")
    stats.append(f"{'='*60}")
    stats.append(f"Original samples: {original_count}")
    stats.append(f"Total DPO pairs: {len(dpo_pairs)}")
    stats.append(f"Average pairs per sample: {len(dpo_pairs) / original_count:.2f}")
    
    # Count by rejected error category
    rejected_error_counts = defaultdict(int)
    for pair in dpo_pairs:
        rejected_error_counts[pair["meta"]["rejected_error_category"]] += 1
    
    stats.append(f"\nRejected response distribution:")
    for error_cat, count in sorted(rejected_error_counts.items()):
        percentage = (count / len(dpo_pairs)) * 100
        stats.append(f"  {error_cat:15s}: {count:6d} ({percentage:5.2f}%)")
    
    # Count by chosen error category
    chosen_error_counts = defaultdict(int)
    for pair in dpo_pairs:
        chosen_error_counts[pair["meta"]["chosen_error_category"]] += 1
    
    stats.append(f"\nChosen response distribution:")
    for error_cat, count in sorted(chosen_error_counts.items()):
        percentage = (count / len(dpo_pairs)) * 100
        stats.append(f"  {error_cat:15s}: {count:6d} ({percentage:5.2f}%)")
    
    # Count pairs per unique audio
    unique_audios = set(pair["audio_path"] for pair in dpo_pairs)
    stats.append(f"\nUnique audio files: {len(unique_audios)}")
    
    return "\n".join(stats)


@hydra.main(
    config_path="../configs/data",
    config_name="06_dpo_dataset",
    version_base=None,
)
def main(cfg: DictConfig):
    """Main function to create DPO dataset."""
    print("=" * 60)
    print("Creating DPO Dataset for Direct Preference Optimization")
    print("=" * 60)
    
    # Convert paths to absolute
    input_train = Path(cfg.input.train_jsonl)
    input_test = Path(cfg.input.test_jsonl)
    output_train = Path(cfg.output.train_dpo_jsonl)
    output_test = Path(cfg.output.test_dpo_jsonl)
    stats_file = Path(cfg.output.stats_file)
    
    all_stats = []
    
    # Process training set
    print(f"\n{'='*60}")
    print("Processing Training Set")
    print(f"{'='*60}")
    print(f"Loading: {input_train}")
    train_data = load_jsonl(input_train)
    print(f"Loaded {len(train_data)} training samples")
    
    train_dpo_pairs = create_dpo_pairs(
        original_data=train_data,
        templates=dict(cfg.response_templates),
        error_ranges=dict(cfg.error_ranges_db),
        error_categories=list(cfg.error_categories),
    )
    
    print(f"Generated {len(train_dpo_pairs)} DPO pairs")
    print(f"Saving to: {output_train}")
    save_jsonl(train_dpo_pairs, output_train)
    
    train_stats = generate_statistics(len(train_data), train_dpo_pairs, "train")
    print(train_stats)
    all_stats.append(train_stats)
    
    # Process test set
    print(f"\n{'='*60}")
    print("Processing Test Set")
    print(f"{'='*60}")
    print(f"Loading: {input_test}")
    test_data = load_jsonl(input_test)
    print(f"Loaded {len(test_data)} test samples")
    
    test_dpo_pairs = create_dpo_pairs(
        original_data=test_data,
        templates=dict(cfg.response_templates),
        error_ranges=dict(cfg.error_ranges_db),
        error_categories=list(cfg.error_categories),
    )
    
    print(f"Generated {len(test_dpo_pairs)} DPO pairs")
    print(f"Saving to: {output_test}")
    save_jsonl(test_dpo_pairs, output_test)
    
    test_stats = generate_statistics(len(test_data), test_dpo_pairs, "test")
    print(test_stats)
    all_stats.append(test_stats)
    
    # Save statistics to file
    print(f"\n{'='*60}")
    print(f"Saving statistics to: {stats_file}")
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w") as f:
        f.write("\n".join(all_stats))
    
    print(f"\n{'='*60}")
    print("DPO Dataset Creation Complete!")
    print(f"{'='*60}")
    print(f"Train DPO pairs: {len(train_dpo_pairs)}")
    print(f"Test DPO pairs: {len(test_dpo_pairs)}")
    print(f"Total DPO pairs: {len(train_dpo_pairs) + len(test_dpo_pairs)}")


if __name__ == "__main__":
    main()

