#!/usr/bin/env python3
"""Fix the no_error template in DPO dataset to be length-balanced."""

import json
from pathlib import Path

# New balanced template (21 words vs old 7 words)
OLD_TEMPLATE = "The mix is well-balanced. No adjustments needed."
NEW_TEMPLATE = "After analyzing the mix, all stems are at appropriate levels and the balance is correct. The mix is well-balanced. No adjustments needed."

def fix_jsonl(input_path: Path, output_path: Path):
    """Replace the no_error template in a JSONL file."""
    fixed_count = 0
    total_count = 0
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            total_count += 1
            data = json.loads(line)
            
            # Fix chosen response if it's the old template
            if data['chosen'] == OLD_TEMPLATE:
                data['chosen'] = NEW_TEMPLATE
                fixed_count += 1
            
            # Fix rejected response if it's the old template
            if data['rejected'] == OLD_TEMPLATE:
                data['rejected'] = NEW_TEMPLATE
                fixed_count += 1
            
            outfile.write(json.dumps(data) + '\n')
    
    return fixed_count, total_count

def main():
    base_dir = Path(__file__).parent
    
    # Fix training dataset
    train_input = base_dir / "data/musdb18hq_processed/train/training_samples_dpo.jsonl"
    train_output = base_dir / "data/musdb18hq_processed/train/training_samples_dpo_fixed.jsonl"
    
    print(f"Fixing training dataset...")
    train_fixed, train_total = fix_jsonl(train_input, train_output)
    print(f"  Fixed {train_fixed} instances in {train_total} pairs")
    
    # Fix test dataset
    test_input = base_dir / "data/musdb18hq_processed/test/test_samples_dpo.jsonl"
    test_output = base_dir / "data/musdb18hq_processed/test/test_samples_dpo_fixed.jsonl"
    
    print(f"Fixing test dataset...")
    test_fixed, test_total = fix_jsonl(test_input, test_output)
    print(f"  Fixed {test_fixed} instances in {test_total} pairs")
    
    # Replace original files
    print("\nReplacing original files...")
    train_input.rename(train_input.with_suffix('.jsonl.old'))
    train_output.rename(train_input)
    
    test_input.rename(test_input.with_suffix('.jsonl.old'))
    test_output.rename(test_input)
    
    print("✅ Done!")
    print(f"Old template: '{OLD_TEMPLATE}' ({len(OLD_TEMPLATE.split())} words)")
    print(f"New template: '{NEW_TEMPLATE}' ({len(NEW_TEMPLATE.split())} words)")

if __name__ == "__main__":
    main()

