#!/usr/bin/env python3
"""
Reduce JSONL files by sampling 1/10 of each category.
"""
import json
import sys
from collections import defaultdict
import random

def reduce_jsonl_by_category(input_file, output_file, reduction_factor=10, seed=42):
    """
    Read a JSONL file, group by error_category, and keep 1/reduction_factor of each category.
    """
    # Group entries by category
    categories = defaultdict(list)
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Get category from meta.error_category
                category = entry.get('meta', {}).get('error_category', 'unknown')
                categories[category].append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}", file=sys.stderr)
                continue
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Sample from each category
    reduced_entries = []
    for category, entries in sorted(categories.items()):
        original_count = len(entries)
        sample_size = max(1, original_count // reduction_factor)
        sampled = random.sample(entries, sample_size)
        reduced_entries.extend(sampled)
        print(f"  {category}: {original_count} -> {sample_size}")
    
    # Shuffle the final result to mix categories
    random.shuffle(reduced_entries)
    
    # Write output
    print(f"Writing {output_file}...")
    with open(output_file, 'w') as f:
        for entry in reduced_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Done! Reduced from {sum(len(v) for v in categories.values())} to {len(reduced_entries)} entries")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python reduce_jsonl_by_category.py <input_file> <output_file> [reduction_factor]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    reduction_factor = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    reduce_jsonl_by_category(input_file, output_file, reduction_factor)

