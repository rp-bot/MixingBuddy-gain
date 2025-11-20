"""
Explanation of what the 56% accuracy was compared against.

The random baseline (15.56%) assumes:
1. Random stem selection: 1/3 = 33.3% chance
   - There are 3 possible stems: vocals, bass, drums
   - If guessing randomly, you'd get it right 1/3 of the time

2. Random direction selection: ~46.7% chance
   - For quiet/very_quiet cases (34% of data): need "increase" 
     → Random guess between increase/decrease = 50% chance
   - For loud/very_loud cases (46% of data): need "decrease"
     → Random guess between increase/decrease = 50% chance  
   - For no_error cases (20% of data): need "no error"
     → Random guess between 3 options (increase/decrease/no_error) = 33% chance
   - Weighted average: (34% × 50%) + (46% × 50%) + (20% × 33%) = 46.7%

3. Combined random chance: 33.3% × 46.7% = 15.56%
   - This is the probability of getting BOTH stem AND direction correct by random guessing

Alternative baselines to consider:
- Stem-only random: 33.3% (just guessing the stem correctly)
- Direction-only random: 46.7% (just guessing the direction correctly)
- Naive baseline (always guess most common): Would need to check what's most common
"""

import json
from collections import Counter

def show_alternative_baselines(filepath):
    """Show different baseline comparisons"""
    
    # Load data
    error_categories = []
    target_stems = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                error_categories.append(item['error_category'])
                target_stems.append(item['target_stem'])
    
    stem_dist = Counter(target_stems)
    error_dist = Counter(error_categories)
    
    print("=" * 70)
    print("BASELINE COMPARISONS")
    print("=" * 70)
    
    print("\n1. RANDOM GUESSING BASELINE (what I used):")
    print("   Assumes: Completely random selection of stem and direction")
    print("   - Random stem: 1/3 = 33.3% (vocals, bass, or drums)")
    print("   - Random direction: ~46.7% (weighted by error category)")
    print("   - Combined: 33.3% × 46.7% = 15.56%")
    print("   → This is what a 'coin flip' would achieve")
    
    print("\n2. STEM-ONLY BASELINE:")
    most_common_stem = stem_dist.most_common(1)[0]
    stem_only_accuracy = most_common_stem[1] / len(target_stems)
    print(f"   If you always guess '{most_common_stem[0]}' (most common):")
    print(f"   → {stem_only_accuracy*100:.1f}% (but direction would be wrong)")
    
    print("\n3. DIRECTION-ONLY BASELINE:")
    quiet_count = error_dist.get('quiet', 0) + error_dist.get('very_quiet', 0)
    loud_count = error_dist.get('loud', 0) + error_dist.get('very_loud', 0)
    no_error_count = error_dist.get('no_error', 0)
    total = len(error_categories)
    
    # If always guess "increase" for quiet, "decrease" for loud, "no_error" for no_error
    direction_only = (quiet_count + loud_count + no_error_count) / total
    print(f"   If you always guess correct direction (but random stem):")
    print(f"   → {direction_only*100:.1f}% (but stem would be wrong)")
    
    print("\n4. MAJORITY CLASS BASELINE:")
    # Most common combination
    most_common_error = max(error_dist.items(), key=lambda x: x[1])
    most_common_stem_name = most_common_stem[0]
    majority_baseline = 0  # Would need to check actual matches
    print(f"   If you always guess '{most_common_stem_name}' + most common error type:")
    print(f"   → Would need to calculate actual matches")
    
    print("\n5. YOUR MODEL'S PERFORMANCE:")
    print("   → 56.00% (both stem AND direction correct)")
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print(f"Random baseline (15.56%): What you'd get by pure guessing")
    print(f"Your model (56.00%): {56.00/15.56:.1f}x better than random")
    print(f"\nThe 56% is compared against the 15.56% random baseline, which")
    print(f"represents the expected accuracy if someone was randomly guessing")
    print(f"both the stem and direction with no knowledge of the audio.")

if __name__ == "__main__":
    filepath = 'outputs/evaluation/qlora-qwen2-7b-mert-musdb-expanded-r16a32-musdb/predictions/predictions.jsonl'
    show_alternative_baselines(filepath)

