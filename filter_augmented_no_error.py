#!/usr/bin/env python3
"""
Filter out augmented 'no_error' samples from the training dataset.
Removes entries where global_uid contains '_aug_' and error_category is 'no_error'.
"""

import json

input_file = "data/musdb18hq_processed/train/training_samples_augmented_variations.jsonl"
output_file = "data/musdb18hq_processed/train/training_samples_augmented_variations_filtered.jsonl"

print(f"Processing {input_file}...")

removed_count = 0
kept_count = 0

with open(input_file, 'r') as f_in, open(output_file + '.tmp', 'w') as f_out:
    for line_num, line in enumerate(f_in, 1):
        if line_num % 10000 == 0:
            print(f"  Processed {line_num} lines... (kept: {kept_count}, removed: {removed_count})")
        
        try:
            data = json.loads(line.strip())
            global_uid = data.get('global_uid', '')
            error_category = data.get('error_category', '')
            
            # Remove if it's an augmented sample with no_error
            if '_aug_' in global_uid and error_category == 'no_error':
                removed_count += 1
                continue
            
            # Keep all other entries
            f_out.write(line)
            kept_count += 1
            
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
            continue

# Rename temp file to final output
import shutil
shutil.move(output_file + '.tmp', output_file)

print(f"\nDone!")
print(f"  Total lines processed: {removed_count + kept_count}")
print(f"  Lines kept: {kept_count}")
print(f"  Lines removed (augmented no_error): {removed_count}")
print(f"  Original file preserved: {input_file}")
print(f"  Filtered file created: {output_file}")

