import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

def analyze_prediction(item):
    """
    Analyzes a single prediction item to check for correctness of stem and direction.
    """
    target_stem = item['target_stem']
    error_category = item['error_category']
    generated_text = item['generated'].lower()
    
    # Define keywords for directions
    increase_keywords = ['increase', 'too quiet', 'barely audible', 'too low']
    decrease_keywords = ['reduce', 'decrease', 'too loud', 'overwhelming', 'too high']
    no_error_keywords = ['no adjustments', 'well-balanced', 'correct level', 'balanced', 'no changes needed']

    # Determine expected direction
    if error_category in ['quiet', 'very_quiet']:
        expected_direction = 'increase'
    elif error_category in ['loud', 'very_loud']:
        expected_direction = 'decrease'
    else: # no_error
        expected_direction = 'none'

    # Normalize target_stem for searching
    stem_map = {
        'vocals': 'vocals',
        'drums': 'drums',
        'bass': 'bass',
        'other': '(other|other instrument)'
    }

    # Split into sentences
    generated_sents = re.split(r'[.!?]', generated_text)
    
    # Check for contradictions
    stems_to_check = ['vocals', 'drums', 'bass', '(other|other instrument)']
    for stem_pattern in stems_to_check:
        stem_sents = [s for s in generated_sents if re.search(r'\b' + stem_pattern + r'\b', s)]
        if not stem_sents:
            continue
        
        has_increase = any(any(kw in s for kw in increase_keywords) for s in stem_sents)
        has_decrease = any(any(kw in s for kw in decrease_keywords) for s in stem_sents)
        
        if has_increase and has_decrease:
            return False, False, False  # Contradictory

    # Identify the main stem being talked about, and the direction
    identified_stem = None
    identified_direction = None

    # Look for sentences that contain a stem and a directional keyword.
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
    
    # If no problem stem was identified, check for 'no_error' case
    if not identified_stem:
        if any(any(kw in s for kw in no_error_keywords) for s in generated_sents):
            identified_direction = 'none'
            identified_stem = target_stem
    
    # Compare with ground truth
    direction_correct = (identified_direction == expected_direction)
    
    # For "no_error" cases, stem_correct should match direction_correct
    # (if model correctly says no adjustment needed, both are correct; otherwise both are wrong)
    if error_category == 'no_error':
        stem_correct = direction_correct
    else:
        stem_correct = (identified_stem == target_stem)
    
    both_correct = stem_correct and direction_correct

    return both_correct, stem_correct, direction_correct

def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = 'outputs/evaluation/qlora-qwen2-7b-mert-top3-musdb-expanded-lora-r16a32-musdb/predictions/predictions.jsonl'
    
    # Statistics
    total = 0
    both_correct = 0
    stem_correct = 0
    direction_correct = 0
    
    # Breakdowns
    by_error_category = defaultdict(lambda: {'total': 0, 'both': 0, 'stem': 0, 'direction': 0})
    by_stem = defaultdict(lambda: {'total': 0, 'both': 0, 'stem': 0, 'direction': 0})
    by_direction = defaultdict(lambda: {'total': 0, 'both': 0, 'stem': 0, 'direction': 0})
    
    # Error analysis
    errors = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                total += 1
                item = json.loads(line)
                
                both, stem, direction = analyze_prediction(item)
                
                if both:
                    both_correct += 1
                if stem:
                    stem_correct += 1
                if direction:
                    direction_correct += 1
                
                # Track by error category
                error_cat = item['error_category']
                by_error_category[error_cat]['total'] += 1
                if both:
                    by_error_category[error_cat]['both'] += 1
                if stem:
                    by_error_category[error_cat]['stem'] += 1
                if direction:
                    by_error_category[error_cat]['direction'] += 1
                
                # Track by stem
                target_stem = item['target_stem']
                by_stem[target_stem]['total'] += 1
                if both:
                    by_stem[target_stem]['both'] += 1
                if stem:
                    by_stem[target_stem]['stem'] += 1
                if direction:
                    by_stem[target_stem]['direction'] += 1
                
                # Track by direction needed
                if error_cat in ['quiet', 'very_quiet']:
                    dir_type = 'increase'
                elif error_cat in ['loud', 'very_loud']:
                    dir_type = 'decrease'
                else:  # no_error
                    dir_type = 'no_error'
                
                by_direction[dir_type]['total'] += 1
                if both:
                    by_direction[dir_type]['both'] += 1
                if stem:
                    by_direction[dir_type]['stem'] += 1
                if direction:
                    by_direction[dir_type]['direction'] += 1
                
                # Collect errors for analysis
                if not both:
                    errors.append({
                        'uid': item.get('global_uid', 'unknown'),
                        'target_stem': target_stem,
                        'error_category': error_cat,
                        'generated': item['generated'][:100]  # First 100 chars
                    })
    
    # Print results
    print("=" * 70)
    print("DETAILED EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Total predictions: {total}")
    print(f"  Both stem AND direction correct: {both_correct} ({both_correct/total*100:.2f}%)")
    print(f"  Stem correct: {stem_correct} ({stem_correct/total*100:.2f}%)")
    print(f"  Direction correct: {direction_correct} ({direction_correct/total*100:.2f}%)")
    
    print(f"\nBREAKDOWN BY ERROR CATEGORY:")
    for error_cat in sorted(by_error_category.keys()):
        stats = by_error_category[error_cat]
        print(f"  {error_cat:15s}: {stats['both']:2d}/{stats['total']:2d} both ({stats['both']/stats['total']*100:5.1f}%) | "
              f"stem: {stats['stem']/stats['total']*100:5.1f}% | "
              f"direction: {stats['direction']/stats['total']*100:5.1f}%")
    
    print(f"\nBREAKDOWN BY TARGET STEM:")
    for stem in sorted(by_stem.keys()):
        stats = by_stem[stem]
        print(f"  {stem:15s}: {stats['both']:2d}/{stats['total']:2d} both ({stats['both']/stats['total']*100:5.1f}%) | "
              f"stem: {stats['stem']/stats['total']*100:5.1f}% | "
              f"direction: {stats['direction']/stats['total']*100:5.1f}%")
    
    print(f"\nBREAKDOWN BY DIRECTION TYPE:")
    for dir_type in sorted(by_direction.keys()):
        stats = by_direction[dir_type]
        print(f"  {dir_type:15s}: {stats['both']:2d}/{stats['total']:2d} both ({stats['both']/stats['total']*100:5.1f}%) | "
              f"stem: {stats['stem']/stats['total']*100:5.1f}% | "
              f"direction: {stats['direction']/stats['total']*100:5.1f}%")
    
    print(f"\nERROR ANALYSIS (first 10 examples where both were wrong):")
    for i, error in enumerate(errors[:10]):
        print(f"  {i+1}. {error['uid']}")
        print(f"     Target: {error['target_stem']} ({error['error_category']})")
        print(f"     Generated: {error['generated']}...")
    
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more errors")
    
    # Prepare results for JSON output
    results = {
        'overall_performance': {
            'total_predictions': total,
            'both_correct': both_correct,
            'both_correct_percentage': round(both_correct/total*100, 2) if total > 0 else 0.0,
            'stem_correct': stem_correct,
            'stem_correct_percentage': round(stem_correct/total*100, 2) if total > 0 else 0.0,
            'direction_correct': direction_correct,
            'direction_correct_percentage': round(direction_correct/total*100, 2) if total > 0 else 0.0
        },
        'by_error_category': {},
        'by_target_stem': {},
        'by_direction_type': {},
        'error_analysis': {
            'total_errors': len(errors),
            'error_examples': errors[:20]  # Include first 20 errors
        }
    }
    
    # Convert defaultdicts to regular dicts and calculate percentages
    for error_cat in sorted(by_error_category.keys()):
        stats = by_error_category[error_cat]
        results['by_error_category'][error_cat] = {
            'total': stats['total'],
            'both_correct': stats['both'],
            'both_correct_percentage': round(stats['both']/stats['total']*100, 2) if stats['total'] > 0 else 0.0,
            'stem_correct': stats['stem'],
            'stem_correct_percentage': round(stats['stem']/stats['total']*100, 2) if stats['total'] > 0 else 0.0,
            'direction_correct': stats['direction'],
            'direction_correct_percentage': round(stats['direction']/stats['total']*100, 2) if stats['total'] > 0 else 0.0
        }
    
    for stem in sorted(by_stem.keys()):
        stats = by_stem[stem]
        results['by_target_stem'][stem] = {
            'total': stats['total'],
            'both_correct': stats['both'],
            'both_correct_percentage': round(stats['both']/stats['total']*100, 2) if stats['total'] > 0 else 0.0,
            'stem_correct': stats['stem'],
            'stem_correct_percentage': round(stats['stem']/stats['total']*100, 2) if stats['total'] > 0 else 0.0,
            'direction_correct': stats['direction'],
            'direction_correct_percentage': round(stats['direction']/stats['total']*100, 2) if stats['total'] > 0 else 0.0
        }
    
    for dir_type in sorted(by_direction.keys()):
        stats = by_direction[dir_type]
        results['by_direction_type'][dir_type] = {
            'total': stats['total'],
            'both_correct': stats['both'],
            'both_correct_percentage': round(stats['both']/stats['total']*100, 2) if stats['total'] > 0 else 0.0,
            'stem_correct': stats['stem'],
            'stem_correct_percentage': round(stats['stem']/stats['total']*100, 2) if stats['total'] > 0 else 0.0,
            'direction_correct': stats['direction'],
            'direction_correct_percentage': round(stats['direction']/stats['total']*100, 2) if stats['total'] > 0 else 0.0
        }
    
    # Save results to JSON file
    predictions_path = Path(filepath)
    output_dir = predictions_path.parent
    output_file = output_dir / 'evaluation_results.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()

