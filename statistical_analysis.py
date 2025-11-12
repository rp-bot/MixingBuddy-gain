import json
import re
from scipy import stats
import numpy as np

def calculate_random_baseline(filepath):
    """
    Calculate the expected random baseline accuracy.
    """
    error_categories = []
    target_stems = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                error_categories.append(item['error_category'])
                target_stems.append(item['target_stem'])
    
    # Count distributions
    from collections import Counter
    error_dist = Counter(error_categories)
    stem_dist = Counter(target_stems)
    
    print("Error category distribution:")
    for cat, count in error_dist.items():
        print(f"  {cat}: {count} ({count/len(error_categories)*100:.1f}%)")
    
    print("\nStem distribution:")
    for stem, count in stem_dist.items():
        print(f"  {stem}: {count} ({count/len(target_stems)*100:.1f}%)")
    
    # Calculate random baseline
    # For stem: 1/4 = 25% (assuming uniform, but we can use actual distribution)
    # For direction: depends on error category
    #   - quiet/very_quiet: need "increase" (50% if random between increase/decrease)
    #   - loud/very_loud: need "decrease" (50% if random between increase/decrease)
    #   - no_error: need "none" (harder to estimate, but let's say 33% if random between 3 options)
    
    # More accurate: if completely random, probability of correct stem = 1/num_stems
    num_stems = len(stem_dist)
    prob_correct_stem = 1 / num_stems
    
    # For direction, if random between increase/decrease, it's 50%
    # But we also need to account for no_error cases
    quiet_count = error_dist.get('quiet', 0) + error_dist.get('very_quiet', 0)
    loud_count = error_dist.get('loud', 0) + error_dist.get('very_loud', 0)
    no_error_count = error_dist.get('no_error', 0)
    total = len(error_categories)
    
    # If random, direction accuracy would be:
    # - For quiet/loud: 50% (random between increase/decrease)
    # - For no_error: let's assume model might say "no error" 33% of the time if random
    prob_correct_direction = (quiet_count + loud_count) / total * 0.5 + no_error_count / total * (1/3)
    
    # Combined probability (stem AND direction both correct)
    random_baseline = prob_correct_stem * prob_correct_direction
    
    print(f"\nRandom baseline calculation:")
    print(f"  P(correct stem) = {prob_correct_stem:.3f} (1/{num_stems})")
    print(f"  P(correct direction) = {prob_correct_direction:.3f}")
    print(f"  Random baseline = {random_baseline:.3f} = {random_baseline*100:.2f}%")
    
    return random_baseline, total

def analyze_actual_accuracy(filepath):
    """
    Calculate actual accuracy using the same logic as analyze_predictions.py
    """
    def analyze_prediction(item):
        target_stem = item['target_stem']
        error_category = item['error_category']
        generated_text = item['generated'].lower()
        
        increase_keywords = ['increase', 'too quiet', 'barely audible', 'too low']
        decrease_keywords = ['reduce', 'decrease', 'too loud', 'overwhelming', 'too high']
        no_error_keywords = ['no adjustments', 'well-balanced', 'correct level', 'balanced', 'no changes needed']

        if error_category in ['quiet', 'very_quiet']:
            expected_direction = 'increase'
        elif error_category in ['loud', 'very_loud']:
            expected_direction = 'decrease'
        else:
            expected_direction = 'none'

        stem_map = {
            'vocals': 'vocals',
            'drums': 'drums',
            'bass': 'bass',
            'other': '(other|other instrument)'
        }

        generated_sents = re.split(r'[.!?]', generated_text)
        
        stems_to_check = ['vocals', 'drums', 'bass', '(other|other instrument)']
        for stem_pattern in stems_to_check:
            stem_sents = [s for s in generated_sents if re.search(r'\b' + stem_pattern + r'\b', s)]
            if not stem_sents:
                continue
            
            has_increase = any(any(kw in s for kw in increase_keywords) for s in stem_sents)
            has_decrease = any(any(kw in s for kw in decrease_keywords) for s in stem_sents)
            
            if has_increase and has_decrease:
                return False

        identified_stem = None
        identified_direction = None

        for sent in generated_sents:
            for stem, pattern in stem_map.items():
                if re.search(r'\b' + pattern + r'\b', sent):
                    if any(kw in sent for kw in increase_keywords):
                        identified_stem = stem
                        identified_direction = 'increase'
                        break
                    if any(kw in sent for kw in decrease_keywords):
                        identified_stem = stem
                        identified_direction = 'decrease'
                        break
            if identified_stem:
                break
        
        if not identified_stem:
            if any(any(kw in s for kw in no_error_keywords) for s in generated_sents):
                identified_direction = 'none'
                identified_stem = target_stem
        
        stem_correct = (identified_stem == target_stem)
        direction_correct = (identified_direction == expected_direction)

        return stem_correct and direction_correct

    correct = 0
    total = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                total += 1
                item = json.loads(line)
                if analyze_prediction(item):
                    correct += 1
    
    return correct, total

def binomial_test(observed_success, total_trials, expected_prob):
    """
    Perform a binomial test to see if observed success rate is significantly different from expected.
    """
    # Two-tailed test: is observed significantly different from expected?
    p_value = stats.binom_test(observed_success, total_trials, expected_prob, alternative='two-sided')
    
    # One-tailed test: is observed significantly better than expected?
    p_value_one_tailed = stats.binom_test(observed_success, total_trials, expected_prob, alternative='greater')
    
    return p_value, p_value_one_tailed

def main():
    filepath = 'outputs/evaluation/qlora-qwen2-7b-mert-musdb-expanded-r16a32-musdb/predictions/predictions.jsonl'
    
    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)
    
    # Calculate random baseline
    random_baseline, total = calculate_random_baseline(filepath)
    
    # Calculate actual accuracy
    correct, total_actual = analyze_actual_accuracy(filepath)
    actual_accuracy = correct / total_actual
    
    print(f"\nActual performance:")
    print(f"  Correct: {correct}/{total_actual}")
    print(f"  Accuracy: {actual_accuracy:.3f} = {actual_accuracy*100:.2f}%")
    
    # Statistical test
    print(f"\nStatistical significance test:")
    print(f"  Null hypothesis: Model performs at random baseline ({random_baseline*100:.2f}%)")
    print(f"  Alternative hypothesis: Model performs better than random")
    
    p_value, p_value_one_tailed = binomial_test(correct, total_actual, random_baseline)
    
    print(f"\n  Two-tailed p-value: {p_value:.6f}")
    print(f"  One-tailed p-value (better than random): {p_value_one_tailed:.6f}")
    
    if p_value_one_tailed < 0.05:
        print(f"\n  ✓ SIGNIFICANT: p < 0.05")
        print(f"    The model performs significantly better than random chance!")
    elif p_value_one_tailed < 0.01:
        print(f"\n  ✓ HIGHLY SIGNIFICANT: p < 0.01")
        print(f"    The model performs highly significantly better than random chance!")
    else:
        print(f"\n  ✗ NOT SIGNIFICANT: p >= 0.05")
        print(f"    Cannot reject null hypothesis that model performs at random.")
    
    # Effect size
    improvement = actual_accuracy - random_baseline
    print(f"\n  Improvement over random: {improvement*100:.2f} percentage points")
    print(f"  Relative improvement: {(improvement/random_baseline)*100:.1f}% better than random")

if __name__ == "__main__":
    main()

