#!/usr/bin/env python3
"""
Create DPO (Direct Preference Optimization) dataset from existing mixing dataset.

For each audio sample with a correct response (chosen), generate all possible
alternative incorrect responses (rejected) based on other stem-error combinations.
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
    use_random_template: bool = False,
) -> str:
    """
    Generate a response for a given stem-error combination.
    
    Args:
        target_stem: The stem to mention in the response (e.g., "vocals", "drums")
        error_category: The error type (e.g., "quiet", "loud", "no_error")
        templates: Response templates from config (can be list of variations)
        error_ranges: dB ranges for each error category
        use_random_template: If True, randomly select from template variations
        
    Returns:
        Generated response string
    """
    # Get template for this error category
    template_list = templates[error_category]
    if use_random_template and len(template_list) > 1:
        template = random.choice(template_list)
    else:
        template = template_list[0]  # Use first template
    
    # Special case: no_error doesn't reference a specific stem
    if error_category == "no_error":
        return template
    
    # Get the dB range for this error category
    error_range = error_ranges[error_category]
    
    # Calculate correction range (opposite of applied error)
    # Note: error_range is [less_severe, more_severe]
    # For quiet: [-3, -6] means from -3 dB (slightly quiet) to -6 dB (very quiet)
    # For loud: [3, 6] means from +3 dB (slightly loud) to +6 dB (very loud)
    if error_category in ["quiet", "very_quiet"]:
        # Error was negative gain, so correction is positive increase
        # For [-3, -6], we want to correct by 3 to 6 dB (more correction for more error)
        # max(error_range) gives -3 (less negative), min(error_range) gives -6 (more negative)
        min_correction = abs(max(error_range))  # abs(-3) = 3 (for less severe error)
        max_correction = abs(min(error_range))  # abs(-6) = 6 (for more severe error)
    else:  # loud, very_loud
        # Error was positive gain, so correction is reduction
        # For [3, 6], we want to reduce by 3 to 6 dB
        min_correction = min(error_range)  # 3 (for less severe error)
        max_correction = max(error_range)  # 6 or 12 (for more severe error)
    
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
    use_random_template: bool = False,
    max_rejected: Optional[int] = None,
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
        use_random_template: If True, randomly select from template variations
        max_rejected: Optional limit on number of rejected responses to generate
        
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
            response = generate_response(None, error_cat, templates, error_ranges, use_random_template)
            alternatives.append((response, "none", error_cat))
        else:
            # Generate for each available stem
            for stem in stems_present:
                # Skip if this is the chosen response
                if stem == chosen_stem and error_cat == chosen_error:
                    continue
                    
                response = generate_response(stem, error_cat, templates, error_ranges, use_random_template)
                alternatives.append((response, stem, error_cat))
    
    # Optionally limit the number of rejected alternatives
    if max_rejected is not None and len(alternatives) > max_rejected:
        alternatives = random.sample(alternatives, max_rejected)
    
    return alternatives


def create_dpo_pairs(
    original_data: List[Dict],
    templates: Dict[str, List[str]],
    error_ranges: Dict[str, List[float]],
    error_categories: List[str],
    use_random_template: bool = False,
    max_rejected_per_sample: Optional[int] = None,
) -> List[Dict]:
    """
    Create DPO preference pairs from original dataset.
    
    Args:
        original_data: List of original training samples
        templates: Response templates from config
        error_ranges: dB ranges for each error category
        error_categories: List of all error categories
        use_random_template: If True, randomly select from template variations
        max_rejected_per_sample: Optional limit on rejected responses per sample
        
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
            use_random_template=use_random_template,
            max_rejected=max_rejected_per_sample,
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
    config_name="09_dpo_dataset_variations",
    version_base=None,
)
def main(cfg: DictConfig):
    """Main function to create DPO dataset."""
    print("=" * 60)
    print("Creating DPO Dataset for Direct Preference Optimization")
    print("=" * 60)
    
    # Set random seed for reproducibility
    seed = cfg.processing.get("seed", 42)
    random.seed(seed)
    print(f"Random seed: {seed}")
    
    # Get processing options
    use_random_template = cfg.processing.get("use_random_template", True)
    max_rejected_per_sample = cfg.processing.get("max_rejected_per_sample", None)
    max_train_samples = cfg.processing.get("max_train_samples", None)
    
    print(f"Use random template: {use_random_template}")
    if max_rejected_per_sample:
        print(f"Max rejected per sample: {max_rejected_per_sample}")
    if max_train_samples:
        print(f"Max training samples: {max_train_samples}")
    
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
    
    # Optionally limit training samples with stratified sampling to maintain distribution
    # no_error should equal the sum of all error categories (quiet + very_quiet + loud + very_loud)
    if max_train_samples is not None and len(train_data) > max_train_samples:
        print(f"Sampling {max_train_samples} samples from {len(train_data)} total")
        print("Using stratified sampling: no_error = sum of all error categories...")
        
        # Group samples by error category
        samples_by_category = defaultdict(list)
        for sample in train_data:
            error_cat = sample["meta"]["error_category"]
            samples_by_category[error_cat].append(sample)
        
        # Separate no_error from error categories
        error_categories = [cat for cat in samples_by_category.keys() if cat != "no_error"]
        num_error_categories = len(error_categories)
        
        if num_error_categories == 0:
            # Only no_error samples, just sample directly
            no_error_samples = samples_by_category.get("no_error", [])
            train_data = random.sample(no_error_samples, min(max_train_samples, len(no_error_samples)))
            print(f"  no_error: {len(train_data)} samples")
        else:
            # Calculate distribution: no_error = sum of all error categories
            # So: no_error_count = error_count_per_category * num_error_categories
            # Total = no_error_count + error_count_per_category * num_error_categories
            # Total = error_count_per_category * (num_error_categories + num_error_categories)
            # error_count_per_category = Total / (2 * num_error_categories)
            error_samples_per_category = max_train_samples // (2 * num_error_categories)
            remainder = max_train_samples % (2 * num_error_categories)
            
            # Distribute remainder: give extra samples to error categories first (round-robin),
            # then adjust no_error to match the total
            sampled_data = []
            error_samples_total = 0
            
            # Sample from error categories (uniform distribution among errors)
            for i, error_cat in enumerate(error_categories):
                category_samples = samples_by_category[error_cat]
                # Distribute remainder to first few error categories
                n_samples = error_samples_per_category + (1 if i < remainder else 0)
                n_samples = min(n_samples, len(category_samples))
                
                if n_samples > 0:
                    sampled = random.sample(category_samples, n_samples)
                    sampled_data.extend(sampled)
                    error_samples_total += len(sampled)
                    print(f"  {error_cat:15s}: {len(sampled):5d} / {len(category_samples):5d} samples")
            
            # Sample from no_error (should equal sum of error categories)
            no_error_samples = samples_by_category.get("no_error", [])
            no_error_count = min(error_samples_total, len(no_error_samples))
            
            if no_error_count > 0:
                sampled_no_error = random.sample(no_error_samples, no_error_count)
                sampled_data.extend(sampled_no_error)
                print(f"  {'no_error':15s}: {len(sampled_no_error):5d} / {len(no_error_samples):5d} samples")
            
            # Shuffle to mix categories
            random.shuffle(sampled_data)
            train_data = sampled_data
            print(f"Final sampled dataset: {len(train_data)} samples")
            print(f"  Distribution: no_error={sum(1 for s in train_data if s['meta']['error_category'] == 'no_error')}, "
                  f"errors={sum(1 for s in train_data if s['meta']['error_category'] != 'no_error')}")
    
    train_dpo_pairs = create_dpo_pairs(
        original_data=train_data,
        templates=dict(cfg.response_templates),
        error_ranges=dict(cfg.error_ranges_db),
        error_categories=list(cfg.error_categories),
        use_random_template=use_random_template,
        max_rejected_per_sample=max_rejected_per_sample,
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
        use_random_template=use_random_template,
        max_rejected_per_sample=max_rejected_per_sample,
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

