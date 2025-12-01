import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

def extract_gain_range(generated_text):
    """
    Extracts a (min_db, max_db) gain range from the generated text.
    Handles patterns like:
      "between 3 and 6 dB", "3 to 6 dB", "3-6 dB"
    Returns (None, None) if no range could be found.
    """
    if not generated_text:
        return None, None

    text = generated_text.lower()

    patterns = [
        r'between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)\s*dB',
        r'(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)\s*dB',
        r'between\s+(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*dB',
        r'(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*dB',
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                min_db = float(m.group(1))
                max_db = float(m.group(2))
                return min_db, max_db
            except (ValueError, IndexError):
                continue

    return None, None

def extract_solution_sentence(generated_text):
    """
    Best-effort extraction of the 'solution' / adjustment part of the response,
    e.g. from:
      "The bass is too quiet. Raise the bass level by 6 to 12 dB to balance the mix."
    we want:
      "Raise the bass level by 6 to 12 dB to balance the mix."
    """
    if not generated_text:
        return ""

    # Split into sentences while keeping things reasonably robust to punctuation issues
    sentences = re.split(r'(?<=[.!?])\s+', generated_text.strip())
    if not sentences:
        return ""

    # Keywords that typically appear in the adjustment/solution sentence
    adjustment_keywords = [
        'increase', 'boost', 'raise', 'add gain', 'bring up', 'up by',
        'reduce', 'decrease', 'lower', 'bring down', 'down by',
        'adjust down', 'brought down'
    ]

    # Prefer the first sentence that clearly contains an adjustment keyword
    for sent in sentences:
        lower = sent.lower()
        if any(kw in lower for kw in adjustment_keywords):
            return sent.strip()

    # Fallbacks: often the second sentence is the solution, otherwise the first
    if len(sentences) >= 2:
        return sentences[1].strip()
    return sentences[0].strip()

def analyze_prediction(item):
    """
    Analyzes a single prediction item to check for correctness of stem and direction.
    """
    target_stem = item['target_stem']
    error_category = item['error_category']
    generated_text = item['generated']
    generated_text_lower = generated_text.lower()
    
    # Define keywords for directions - updated to match new response template variations
    # Increase keywords (for quiet/very_quiet) - extracted from all 10 template variations
    increase_keywords = [
        'increase', 'boost', 'raise', 'louder', 'quiet', 'too quiet', 'barely audible', 
        'too low', 'needs to be slightly louder', 'somewhat quiet', 'a bit too low',
        'requires a slight boost', 'slightly quiet', 'needs more volume', 'a little low',
        'a little too quiet', 'should be slightly louder', 'somewhat too quiet', 
        'too quiet to hear clearly', 'nearly inaudible', 'too low in the mix', 
        'difficult to hear', 'almost inaudible', 'barely present in the mix', 
        'needs significant boosting', 'too quiet to be heard properly', 'add gain',
        'bring up', 'up by'
    ]
    
    # Decrease keywords (for loud/very_loud) - extracted from all 10 template variations
    decrease_keywords = [
        'reduce', 'decrease', 'lower', 'quieter', 'loud', 'too loud', 'overwhelming', 
        'too high', 'a little too loud', 'needs to be slightly quieter', 'somewhat loud', 
        'a bit too high', 'requires a slight reduction', 'slightly loud', 'needs less volume', 
        'a little high', 'should be slightly quieter', 'somewhat too loud', 
        'too loud and dominating', 'overpowering the mix', 'too high in the mix', 
        'excessively loud', 'dominating the mix', 'overpowering', 'needs significant reduction', 
        'too loud and needs to be brought down', 'higher than it should be', 'should be lower',
        'adjust down', 'bring down', 'down by', 'brought down'
    ]
    
    # No error keywords - extracted from all 10 template variations
    no_error_keywords = [
        'no adjustments', 'no changes', 'no modifications', 'well-balanced', 'correct level',
        'balanced', 'properly balanced', 'proper balance', 'correctly balanced', 'balance is correct',
        'all stems are at appropriate levels', 'all stems are properly balanced',
        'all stems are correctly balanced', 'all stems are at proper levels', 'all stems at proper levels',
        'all stems are at correct levels', 'levels are correct', 'balance is good', 
        'mix is well-balanced', 'mix balance is correct', 'mix evaluation shows proper balance',
        'mix analysis shows', 'mix analysis indicates', 'upon review', 'after evaluation',
        'reviewing the mix', 'after careful analysis', 'after analyzing the mix',
        'proper balance across all stems', 'each element is at the correct level',
        'the mix levels are appropriate', 'no changes are needed', 'no adjustments required',
        'no modifications needed', 'no adjustments are necessary', 'no changes required',
        'no modifications are necessary'
    ]

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
    generated_sents = re.split(r'[.!?]', generated_text_lower)
    
    # Check for contradictions
    stems_to_check = ['vocals', 'drums', 'bass', '(other|other instrument)']
    for stem_pattern in stems_to_check:
        stem_sents = [s for s in generated_sents if re.search(r'\b' + stem_pattern + r'\b', s)]
        if not stem_sents:
            continue
        
        has_increase = any(any(kw in s for kw in increase_keywords) for s in stem_sents)
        has_decrease = any(any(kw in s for kw in decrease_keywords) for s in stem_sents)
        
        if has_increase and has_decrease:
            # Contradictory instructions for the same stem: treat all as incorrect
            return False, False, False, False, None, None  # both_correct, stem_correct, direction_correct, magnitude_correct, identified_stem, identified_direction

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

    # Magnitude (gain range) correctness
    expected_min = item.get('expected_magnitude_min_db')
    expected_max = item.get('expected_magnitude_max_db')
    parsed_min, parsed_max = extract_gain_range(generated_text)

    if (
        expected_min is None or expected_max is None or
        parsed_min is None or parsed_max is None
    ):
        magnitude_correct = False
    else:
        eps = 1e-3
        magnitude_correct = (
            abs(parsed_min - expected_min) <= eps and
            abs(parsed_max - expected_max) <= eps
        )

    return both_correct, stem_correct, direction_correct, magnitude_correct, identified_stem, identified_direction

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
    magnitude_correct = 0
    total_magnitude_eligible = 0  # Excludes no_error cases
    
    # False positives and false negatives
    # Error detection: FP = predicted error when no_error, FN = predicted no_error when error exists
    error_detection_fp = 0  # predicted adjustment when ground truth is no_error
    error_detection_fn = 0  # predicted no_error when ground truth requires adjustment
    error_detection_tp = 0  # predicted adjustment when adjustment is needed
    error_detection_tn = 0  # predicted no_error when no_error is correct
    
    # Stem identification: FP = predicted wrong stem, FN = failed to identify correct stem
    stem_fp = 0  # predicted wrong stem (when a different stem is correct)
    stem_fn = 0  # failed to identify correct stem (predicted different stem or no stem)
    stem_tp = 0  # predicted correct stem
    stem_tn = 0  # N/A for stem (always a stem should be identified if error exists)
    
    # Direction: FP = predicted wrong direction, FN = failed to predict correct direction
    direction_fp = 0  # predicted wrong direction (e.g., increase when decrease needed)
    direction_fn = 0  # failed to predict correct direction (predicted none or wrong direction)
    direction_tp = 0  # predicted correct direction
    direction_tn = 0  # predicted none when none is correct
    
    # Breakdowns
    by_error_category = defaultdict(lambda: {'total': 0, 'both': 0, 'stem': 0, 'direction': 0, 'magnitude': 0, 'magnitude_eligible': 0,
                                             'error_detection_fp': 0, 'error_detection_fn': 0, 'error_detection_tp': 0, 'error_detection_tn': 0,
                                             'stem_fp': 0, 'stem_fn': 0, 'stem_tp': 0,
                                             'direction_fp': 0, 'direction_fn': 0, 'direction_tp': 0, 'direction_tn': 0})
    by_stem = defaultdict(lambda: {'total': 0, 'both': 0, 'stem': 0, 'direction': 0, 'magnitude': 0, 'magnitude_eligible': 0,
                                  'error_detection_fp': 0, 'error_detection_fn': 0, 'error_detection_tp': 0, 'error_detection_tn': 0,
                                  'stem_fp': 0, 'stem_fn': 0, 'stem_tp': 0,
                                  'direction_fp': 0, 'direction_fn': 0, 'direction_tp': 0, 'direction_tn': 0})
    by_direction = defaultdict(lambda: {'total': 0, 'both': 0, 'stem': 0, 'direction': 0, 'magnitude': 0, 'magnitude_eligible': 0,
                                        'error_detection_fp': 0, 'error_detection_fn': 0, 'error_detection_tp': 0, 'error_detection_tn': 0,
                                        'stem_fp': 0, 'stem_fn': 0, 'stem_tp': 0,
                                        'direction_fp': 0, 'direction_fn': 0, 'direction_tp': 0, 'direction_tn': 0})
    
    # Error analysis
    errors = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                total += 1
                item = json.loads(line)
                
                both, stem, direction, magnitude, identified_stem, identified_direction = analyze_prediction(item)
                
                if both:
                    both_correct += 1
                if stem:
                    stem_correct += 1
                if direction:
                    direction_correct += 1
                
                # Calculate false positives and false negatives
                error_cat = item['error_category']
                target_stem = item['target_stem']
                
                # Determine expected direction
                if error_cat in ['quiet', 'very_quiet']:
                    expected_direction = 'increase'
                elif error_cat in ['loud', 'very_loud']:
                    expected_direction = 'decrease'
                else:  # no_error
                    expected_direction = 'none'
                
                # Error detection FP/FN
                predicted_has_error = (identified_direction is not None and identified_direction != 'none')
                ground_truth_has_error = (error_cat != 'no_error')
                
                if predicted_has_error and not ground_truth_has_error:
                    error_detection_fp += 1
                elif not predicted_has_error and ground_truth_has_error:
                    error_detection_fn += 1
                elif predicted_has_error and ground_truth_has_error:
                    error_detection_tp += 1
                elif not predicted_has_error and not ground_truth_has_error:
                    error_detection_tn += 1
                
                # Stem identification FP/FN (only for cases where error exists)
                if ground_truth_has_error:
                    if identified_stem == target_stem:
                        stem_tp += 1
                    elif identified_stem is not None:
                        stem_fp += 1  # predicted wrong stem
                    else:
                        stem_fn += 1  # failed to identify any stem
                
                # Direction FP/FN
                if identified_direction == expected_direction:
                    if expected_direction == 'none':
                        direction_tn += 1
                    else:
                        direction_tp += 1
                elif identified_direction is None:
                    direction_fn += 1  # failed to predict direction
                elif expected_direction == 'none' and identified_direction != 'none':
                    direction_fp += 1  # predicted direction when none needed
                elif expected_direction != 'none' and identified_direction == 'none':
                    direction_fn += 1  # predicted none when direction needed
                elif expected_direction != 'none' and identified_direction != 'none' and identified_direction != expected_direction:
                    direction_fp += 1  # predicted wrong direction
                
                # Track by error category
                by_error_category[error_cat]['total'] += 1
                if both:
                    by_error_category[error_cat]['both'] += 1
                if stem:
                    by_error_category[error_cat]['stem'] += 1
                if direction:
                    by_error_category[error_cat]['direction'] += 1
                
                # Track FP/FN by error category
                if predicted_has_error and not ground_truth_has_error:
                    by_error_category[error_cat]['error_detection_fp'] += 1
                elif not predicted_has_error and ground_truth_has_error:
                    by_error_category[error_cat]['error_detection_fn'] += 1
                elif predicted_has_error and ground_truth_has_error:
                    by_error_category[error_cat]['error_detection_tp'] += 1
                elif not predicted_has_error and not ground_truth_has_error:
                    by_error_category[error_cat]['error_detection_tn'] += 1
                
                if ground_truth_has_error:
                    if identified_stem == target_stem:
                        by_error_category[error_cat]['stem_tp'] += 1
                    elif identified_stem is not None:
                        by_error_category[error_cat]['stem_fp'] += 1
                    else:
                        by_error_category[error_cat]['stem_fn'] += 1
                
                if identified_direction == expected_direction:
                    if expected_direction == 'none':
                        by_error_category[error_cat]['direction_tn'] += 1
                    else:
                        by_error_category[error_cat]['direction_tp'] += 1
                elif identified_direction is None:
                    by_error_category[error_cat]['direction_fn'] += 1
                elif expected_direction == 'none' and identified_direction != 'none':
                    by_error_category[error_cat]['direction_fp'] += 1
                elif expected_direction != 'none' and identified_direction == 'none':
                    by_error_category[error_cat]['direction_fn'] += 1
                elif expected_direction != 'none' and identified_direction != 'none' and identified_direction != expected_direction:
                    by_error_category[error_cat]['direction_fp'] += 1
                
                # Only count magnitude for non-no_error cases
                if error_cat != 'no_error':
                    total_magnitude_eligible += 1
                    by_error_category[error_cat]['magnitude_eligible'] += 1
                    if magnitude:
                        magnitude_correct += 1
                        by_error_category[error_cat]['magnitude'] += 1
                
                # Track by stem
                by_stem[target_stem]['total'] += 1
                if both:
                    by_stem[target_stem]['both'] += 1
                if stem:
                    by_stem[target_stem]['stem'] += 1
                if direction:
                    by_stem[target_stem]['direction'] += 1
                
                # Track FP/FN by stem
                if predicted_has_error and not ground_truth_has_error:
                    by_stem[target_stem]['error_detection_fp'] += 1
                elif not predicted_has_error and ground_truth_has_error:
                    by_stem[target_stem]['error_detection_fn'] += 1
                elif predicted_has_error and ground_truth_has_error:
                    by_stem[target_stem]['error_detection_tp'] += 1
                elif not predicted_has_error and not ground_truth_has_error:
                    by_stem[target_stem]['error_detection_tn'] += 1
                
                if ground_truth_has_error:
                    if identified_stem == target_stem:
                        by_stem[target_stem]['stem_tp'] += 1
                    elif identified_stem is not None:
                        by_stem[target_stem]['stem_fp'] += 1
                    else:
                        by_stem[target_stem]['stem_fn'] += 1
                
                if identified_direction == expected_direction:
                    if expected_direction == 'none':
                        by_stem[target_stem]['direction_tn'] += 1
                    else:
                        by_stem[target_stem]['direction_tp'] += 1
                elif identified_direction is None:
                    by_stem[target_stem]['direction_fn'] += 1
                elif expected_direction == 'none' and identified_direction != 'none':
                    by_stem[target_stem]['direction_fp'] += 1
                elif expected_direction != 'none' and identified_direction == 'none':
                    by_stem[target_stem]['direction_fn'] += 1
                elif expected_direction != 'none' and identified_direction != 'none' and identified_direction != expected_direction:
                    by_stem[target_stem]['direction_fp'] += 1
                
                # Only count magnitude for non-no_error cases
                if error_cat != 'no_error':
                    by_stem[target_stem]['magnitude_eligible'] += 1
                    if magnitude:
                        by_stem[target_stem]['magnitude'] += 1
                
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
                
                # Track FP/FN by direction
                if predicted_has_error and not ground_truth_has_error:
                    by_direction[dir_type]['error_detection_fp'] += 1
                elif not predicted_has_error and ground_truth_has_error:
                    by_direction[dir_type]['error_detection_fn'] += 1
                elif predicted_has_error and ground_truth_has_error:
                    by_direction[dir_type]['error_detection_tp'] += 1
                elif not predicted_has_error and not ground_truth_has_error:
                    by_direction[dir_type]['error_detection_tn'] += 1
                
                if ground_truth_has_error:
                    if identified_stem == target_stem:
                        by_direction[dir_type]['stem_tp'] += 1
                    elif identified_stem is not None:
                        by_direction[dir_type]['stem_fp'] += 1
                    else:
                        by_direction[dir_type]['stem_fn'] += 1
                
                if identified_direction == expected_direction:
                    if expected_direction == 'none':
                        by_direction[dir_type]['direction_tn'] += 1
                    else:
                        by_direction[dir_type]['direction_tp'] += 1
                elif identified_direction is None:
                    by_direction[dir_type]['direction_fn'] += 1
                elif expected_direction == 'none' and identified_direction != 'none':
                    by_direction[dir_type]['direction_fp'] += 1
                elif expected_direction != 'none' and identified_direction == 'none':
                    by_direction[dir_type]['direction_fn'] += 1
                elif expected_direction != 'none' and identified_direction != 'none' and identified_direction != expected_direction:
                    by_direction[dir_type]['direction_fp'] += 1
                
                # Only count magnitude for non-no_error cases
                if error_cat != 'no_error':
                    by_direction[dir_type]['magnitude_eligible'] += 1
                    if magnitude:
                        by_direction[dir_type]['magnitude'] += 1
                
                # Collect errors for analysis
                if not both:
                    pred_min_db, pred_max_db = extract_gain_range(item['generated'])
                    errors.append({
                        'uid': item.get('global_uid', 'unknown'),
                        'target_stem': target_stem,
                        'error_category': error_cat,
                        'generated': item['generated'][:100],  # First 100 chars
                        'solution_sentence': extract_solution_sentence(item['generated']),
                        'expected_min_db': item.get('expected_magnitude_min_db'),
                        'expected_max_db': item.get('expected_magnitude_max_db'),
                        'predicted_min_db': pred_min_db,
                        'predicted_max_db': pred_max_db,
                        'magnitude_correct': magnitude,
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
    if total_magnitude_eligible > 0:
        print(f"  Magnitude (gain range) correct: {magnitude_correct}/{total_magnitude_eligible} ({magnitude_correct/total_magnitude_eligible*100:.2f}%)")
    else:
        print(f"  Magnitude (gain range) correct: N/A (no eligible cases)")
    
    print(f"\nFALSE POSITIVES AND FALSE NEGATIVES:")
    print(f"  Error Detection:")
    print(f"    True Positives (TP): {error_detection_tp} (predicted error when error exists)")
    print(f"    True Negatives (TN): {error_detection_tn} (predicted no_error when no_error)")
    print(f"    False Positives (FP): {error_detection_fp} (predicted error when no_error)")
    print(f"    False Negatives (FN): {error_detection_fn} (predicted no_error when error exists)")
    if (error_detection_tp + error_detection_fp) > 0:
        precision = error_detection_tp / (error_detection_tp + error_detection_fp) * 100
        print(f"    Precision: {precision:.2f}%")
    if (error_detection_tp + error_detection_fn) > 0:
        recall = error_detection_tp / (error_detection_tp + error_detection_fn) * 100
        print(f"    Recall: {recall:.2f}%")
    if (error_detection_tp + error_detection_fp) > 0 and (error_detection_tp + error_detection_fn) > 0:
        f1 = 2 * precision * recall / (precision + recall)
        print(f"    F1 Score: {f1:.2f}%")
    
    print(f"  Stem Identification (for error cases only):")
    print(f"    True Positives (TP): {stem_tp} (predicted correct stem)")
    print(f"    False Positives (FP): {stem_fp} (predicted wrong stem)")
    print(f"    False Negatives (FN): {stem_fn} (failed to identify any stem)")
    if (stem_tp + stem_fp) > 0:
        precision = stem_tp / (stem_tp + stem_fp) * 100
        print(f"    Precision: {precision:.2f}%")
    if (stem_tp + stem_fn) > 0:
        recall = stem_tp / (stem_tp + stem_fn) * 100
        print(f"    Recall: {recall:.2f}%")
    if (stem_tp + stem_fp) > 0 and (stem_tp + stem_fn) > 0:
        f1 = 2 * precision * recall / (precision + recall)
        print(f"    F1 Score: {f1:.2f}%")
    
    print(f"  Direction Prediction:")
    print(f"    True Positives (TP): {direction_tp} (predicted correct direction)")
    print(f"    True Negatives (TN): {direction_tn} (predicted none when none needed)")
    print(f"    False Positives (FP): {direction_fp} (predicted wrong direction)")
    print(f"    False Negatives (FN): {direction_fn} (failed to predict correct direction)")
    if (direction_tp + direction_fp) > 0:
        precision = direction_tp / (direction_tp + direction_fp) * 100
        print(f"    Precision: {precision:.2f}%")
    if (direction_tp + direction_fn) > 0:
        recall = direction_tp / (direction_tp + direction_fn) * 100
        print(f"    Recall: {recall:.2f}%")
    if (direction_tp + direction_fp) > 0 and (direction_tp + direction_fn) > 0:
        f1 = 2 * precision * recall / (precision + recall)
        print(f"    F1 Score: {f1:.2f}%")
    
    print(f"\nBREAKDOWN BY ERROR CATEGORY:")
    for error_cat in sorted(by_error_category.keys()):
        stats = by_error_category[error_cat]
        mag_str = "N/A"
        if stats['magnitude_eligible'] > 0:
            mag_str = f"{stats['magnitude']/stats['magnitude_eligible']*100:5.1f}%"
        print(f"  {error_cat:15s}: {stats['both']:2d}/{stats['total']:2d} both ({stats['both']/stats['total']*100:5.1f}%) | "
              f"stem: {stats['stem']/stats['total']*100:5.1f}% | "
              f"direction: {stats['direction']/stats['total']*100:5.1f}% | "
              f"magnitude: {mag_str}")
    
    print(f"\nBREAKDOWN BY TARGET STEM:")
    for stem in sorted(by_stem.keys()):
        stats = by_stem[stem]
        mag_str = "N/A"
        if stats['magnitude_eligible'] > 0:
            mag_str = f"{stats['magnitude']/stats['magnitude_eligible']*100:5.1f}%"
        print(f"  {stem:15s}: {stats['both']:2d}/{stats['total']:2d} both ({stats['both']/stats['total']*100:5.1f}%) | "
              f"stem: {stats['stem']/stats['total']*100:5.1f}% | "
              f"direction: {stats['direction']/stats['total']*100:5.1f}% | "
              f"magnitude: {mag_str}")
    
    print(f"\nBREAKDOWN BY DIRECTION TYPE:")
    for dir_type in sorted(by_direction.keys()):
        stats = by_direction[dir_type]
        mag_str = "N/A"
        if stats['magnitude_eligible'] > 0:
            mag_str = f"{stats['magnitude']/stats['magnitude_eligible']*100:5.1f}%"
        print(f"  {dir_type:15s}: {stats['both']:2d}/{stats['total']:2d} both ({stats['both']/stats['total']*100:5.1f}%) | "
              f"stem: {stats['stem']/stats['total']*100:5.1f}% | "
              f"direction: {stats['direction']/stats['total']*100:5.1f}% | "
              f"magnitude: {mag_str}")
    
    print(f"\nERROR ANALYSIS (first 10 examples where both were wrong):")
    for i, error in enumerate(errors[:10]):
        print(f"  {i+1}. {error['uid']}")
        print(f"     Target: {error['target_stem']} ({error['error_category']})")
        print(f"     Generated: {error['generated']}...")
        if error.get('solution_sentence'):
            print(f"     Solution: {error['solution_sentence']}...")
        pred_min = error.get('predicted_min_db')
        pred_max = error.get('predicted_max_db')
        exp_min = error.get('expected_min_db')
        exp_max = error.get('expected_max_db')
        if None not in (pred_min, pred_max, exp_min, exp_max):
            print(
                f"     Gain range (pred vs exp): "
                f"{pred_min:.1f}–{pred_max:.1f} dB "
                f"vs {exp_min:.1f}–{exp_max:.1f} dB "
                f"(magnitude_correct={error['magnitude_correct']})"
            )
    
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more errors")
    
    # Calculate precision, recall, and F1 for JSON output
    error_precision = round(error_detection_tp / (error_detection_tp + error_detection_fp) * 100, 2) if (error_detection_tp + error_detection_fp) > 0 else None
    error_recall = round(error_detection_tp / (error_detection_tp + error_detection_fn) * 100, 2) if (error_detection_tp + error_detection_fn) > 0 else None
    error_f1 = round(2 * error_precision * error_recall / (error_precision + error_recall), 2) if error_precision is not None and error_recall is not None and (error_precision + error_recall) > 0 else None
    
    stem_precision = round(stem_tp / (stem_tp + stem_fp) * 100, 2) if (stem_tp + stem_fp) > 0 else None
    stem_recall = round(stem_tp / (stem_tp + stem_fn) * 100, 2) if (stem_tp + stem_fn) > 0 else None
    stem_f1 = round(2 * stem_precision * stem_recall / (stem_precision + stem_recall), 2) if stem_precision is not None and stem_recall is not None and (stem_precision + stem_recall) > 0 else None
    
    direction_precision = round(direction_tp / (direction_tp + direction_fp) * 100, 2) if (direction_tp + direction_fp) > 0 else None
    direction_recall = round(direction_tp / (direction_tp + direction_fn) * 100, 2) if (direction_tp + direction_fn) > 0 else None
    direction_f1 = round(2 * direction_precision * direction_recall / (direction_precision + direction_recall), 2) if direction_precision is not None and direction_recall is not None and (direction_precision + direction_recall) > 0 else None
    
    # Prepare results for JSON output
    results = {
        'overall_performance': {
            'total_predictions': total,
            'both_correct': both_correct,
            'both_correct_percentage': round(both_correct/total*100, 2) if total > 0 else 0.0,
            'stem_correct': stem_correct,
            'stem_correct_percentage': round(stem_correct/total*100, 2) if total > 0 else 0.0,
            'direction_correct': direction_correct,
            'direction_correct_percentage': round(direction_correct/total*100, 2) if total > 0 else 0.0,
            'magnitude_correct': magnitude_correct,
            'magnitude_eligible': total_magnitude_eligible,
            'magnitude_correct_percentage': round(magnitude_correct/total_magnitude_eligible*100, 2) if total_magnitude_eligible > 0 else None
        },
        'false_positives_negatives': {
            'error_detection': {
                'true_positives': error_detection_tp,
                'true_negatives': error_detection_tn,
                'false_positives': error_detection_fp,
                'false_negatives': error_detection_fn,
                'precision': error_precision,
                'recall': error_recall,
                'f1_score': error_f1
            },
            'stem_identification': {
                'true_positives': stem_tp,
                'false_positives': stem_fp,
                'false_negatives': stem_fn,
                'precision': stem_precision,
                'recall': stem_recall,
                'f1_score': stem_f1
            },
            'direction_prediction': {
                'true_positives': direction_tp,
                'true_negatives': direction_tn,
                'false_positives': direction_fp,
                'false_negatives': direction_fn,
                'precision': direction_precision,
                'recall': direction_recall,
                'f1_score': direction_f1
            }
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
            'direction_correct_percentage': round(stats['direction']/stats['total']*100, 2) if stats['total'] > 0 else 0.0,
            'magnitude_correct': stats['magnitude'],
            'magnitude_eligible': stats['magnitude_eligible'],
            'magnitude_correct_percentage': round(stats['magnitude']/stats['magnitude_eligible']*100, 2) if stats['magnitude_eligible'] > 0 else None,
            'error_detection': {
                'true_positives': stats['error_detection_tp'],
                'true_negatives': stats['error_detection_tn'],
                'false_positives': stats['error_detection_fp'],
                'false_negatives': stats['error_detection_fn']
            },
            'stem_identification': {
                'true_positives': stats['stem_tp'],
                'false_positives': stats['stem_fp'],
                'false_negatives': stats['stem_fn']
            },
            'direction_prediction': {
                'true_positives': stats['direction_tp'],
                'true_negatives': stats['direction_tn'],
                'false_positives': stats['direction_fp'],
                'false_negatives': stats['direction_fn']
            }
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
            'direction_correct_percentage': round(stats['direction']/stats['total']*100, 2) if stats['total'] > 0 else 0.0,
            'magnitude_correct': stats['magnitude'],
            'magnitude_eligible': stats['magnitude_eligible'],
            'magnitude_correct_percentage': round(stats['magnitude']/stats['magnitude_eligible']*100, 2) if stats['magnitude_eligible'] > 0 else None,
            'error_detection': {
                'true_positives': stats['error_detection_tp'],
                'true_negatives': stats['error_detection_tn'],
                'false_positives': stats['error_detection_fp'],
                'false_negatives': stats['error_detection_fn']
            },
            'stem_identification': {
                'true_positives': stats['stem_tp'],
                'false_positives': stats['stem_fp'],
                'false_negatives': stats['stem_fn']
            },
            'direction_prediction': {
                'true_positives': stats['direction_tp'],
                'true_negatives': stats['direction_tn'],
                'false_positives': stats['direction_fp'],
                'false_negatives': stats['direction_fn']
            }
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
            'direction_correct_percentage': round(stats['direction']/stats['total']*100, 2) if stats['total'] > 0 else 0.0,
            'magnitude_correct': stats['magnitude'],
            'magnitude_eligible': stats['magnitude_eligible'],
            'magnitude_correct_percentage': round(stats['magnitude']/stats['magnitude_eligible']*100, 2) if stats['magnitude_eligible'] > 0 else None,
            'error_detection': {
                'true_positives': stats['error_detection_tp'],
                'true_negatives': stats['error_detection_tn'],
                'false_positives': stats['error_detection_fp'],
                'false_negatives': stats['error_detection_fn']
            },
            'stem_identification': {
                'true_positives': stats['stem_tp'],
                'false_positives': stats['stem_fp'],
                'false_negatives': stats['stem_fn']
            },
            'direction_prediction': {
                'true_positives': stats['direction_tp'],
                'true_negatives': stats['direction_tn'],
                'false_positives': stats['direction_fp'],
                'false_negatives': stats['direction_fn']
            }
        }
    
    # Calculate macro accuracies (average across all classes, giving equal weight to each class)
    macro_accuracies = {}
    
    # Macro accuracy by error category
    error_cat_accuracies_both = [results['by_error_category'][cat]['both_correct_percentage'] 
                                 for cat in results['by_error_category'].keys()]
    error_cat_accuracies_stem = [results['by_error_category'][cat]['stem_correct_percentage'] 
                                 for cat in results['by_error_category'].keys()]
    error_cat_accuracies_direction = [results['by_error_category'][cat]['direction_correct_percentage'] 
                                      for cat in results['by_error_category'].keys()]
    error_cat_accuracies_magnitude = [results['by_error_category'][cat]['magnitude_correct_percentage'] 
                                     for cat in results['by_error_category'].keys()
                                     if results['by_error_category'][cat]['magnitude_correct_percentage'] is not None]
    
    macro_accuracies['by_error_category'] = {
        'both_correct_macro': round(sum(error_cat_accuracies_both) / len(error_cat_accuracies_both), 2) if error_cat_accuracies_both else 0.0,
        'stem_correct_macro': round(sum(error_cat_accuracies_stem) / len(error_cat_accuracies_stem), 2) if error_cat_accuracies_stem else 0.0,
        'direction_correct_macro': round(sum(error_cat_accuracies_direction) / len(error_cat_accuracies_direction), 2) if error_cat_accuracies_direction else 0.0,
        'magnitude_correct_macro': round(sum(error_cat_accuracies_magnitude) / len(error_cat_accuracies_magnitude), 2) if error_cat_accuracies_magnitude else None
    }
    
    # Macro accuracy by target stem
    stem_accuracies_both = [results['by_target_stem'][stem]['both_correct_percentage'] 
                           for stem in results['by_target_stem'].keys()]
    stem_accuracies_stem = [results['by_target_stem'][stem]['stem_correct_percentage'] 
                           for stem in results['by_target_stem'].keys()]
    stem_accuracies_direction = [results['by_target_stem'][stem]['direction_correct_percentage'] 
                                for stem in results['by_target_stem'].keys()]
    stem_accuracies_magnitude = [results['by_target_stem'][stem]['magnitude_correct_percentage'] 
                                 for stem in results['by_target_stem'].keys()
                                 if results['by_target_stem'][stem]['magnitude_correct_percentage'] is not None]
    
    macro_accuracies['by_target_stem'] = {
        'both_correct_macro': round(sum(stem_accuracies_both) / len(stem_accuracies_both), 2) if stem_accuracies_both else 0.0,
        'stem_correct_macro': round(sum(stem_accuracies_stem) / len(stem_accuracies_stem), 2) if stem_accuracies_stem else 0.0,
        'direction_correct_macro': round(sum(stem_accuracies_direction) / len(stem_accuracies_direction), 2) if stem_accuracies_direction else 0.0,
        'magnitude_correct_macro': round(sum(stem_accuracies_magnitude) / len(stem_accuracies_magnitude), 2) if stem_accuracies_magnitude else None
    }
    
    # Macro accuracy by direction type
    direction_accuracies_both = [results['by_direction_type'][dir_type]['both_correct_percentage'] 
                                for dir_type in results['by_direction_type'].keys()]
    direction_accuracies_stem = [results['by_direction_type'][dir_type]['stem_correct_percentage'] 
                                for dir_type in results['by_direction_type'].keys()]
    direction_accuracies_direction = [results['by_direction_type'][dir_type]['direction_correct_percentage'] 
                                     for dir_type in results['by_direction_type'].keys()]
    direction_accuracies_magnitude = [results['by_direction_type'][dir_type]['magnitude_correct_percentage'] 
                                      for dir_type in results['by_direction_type'].keys()
                                      if results['by_direction_type'][dir_type]['magnitude_correct_percentage'] is not None]
    
    macro_accuracies['by_direction_type'] = {
        'both_correct_macro': round(sum(direction_accuracies_both) / len(direction_accuracies_both), 2) if direction_accuracies_both else 0.0,
        'stem_correct_macro': round(sum(direction_accuracies_stem) / len(direction_accuracies_stem), 2) if direction_accuracies_stem else 0.0,
        'direction_correct_macro': round(sum(direction_accuracies_direction) / len(direction_accuracies_direction), 2) if direction_accuracies_direction else 0.0,
        'magnitude_correct_macro': round(sum(direction_accuracies_magnitude) / len(direction_accuracies_magnitude), 2) if direction_accuracies_magnitude else None
    }
    
    # Add macro accuracies to results
    results['macro_accuracies'] = macro_accuracies
    
    # Print macro accuracies
    print(f"\nMACRO ACCURACIES (average across all classes):")
    print(f"  By Error Category:")
    print(f"    Both correct: {macro_accuracies['by_error_category']['both_correct_macro']:.2f}%")
    print(f"    Stem correct: {macro_accuracies['by_error_category']['stem_correct_macro']:.2f}%")
    print(f"    Direction correct: {macro_accuracies['by_error_category']['direction_correct_macro']:.2f}%")
    mag_macro = macro_accuracies['by_error_category']['magnitude_correct_macro']
    print(f"    Magnitude correct: {mag_macro:.2f}%" if mag_macro is not None else "    Magnitude correct: N/A")
    print(f"  By Target Stem:")
    print(f"    Both correct: {macro_accuracies['by_target_stem']['both_correct_macro']:.2f}%")
    print(f"    Stem correct: {macro_accuracies['by_target_stem']['stem_correct_macro']:.2f}%")
    print(f"    Direction correct: {macro_accuracies['by_target_stem']['direction_correct_macro']:.2f}%")
    mag_macro = macro_accuracies['by_target_stem']['magnitude_correct_macro']
    print(f"    Magnitude correct: {mag_macro:.2f}%" if mag_macro is not None else "    Magnitude correct: N/A")
    print(f"  By Direction Type:")
    print(f"    Both correct: {macro_accuracies['by_direction_type']['both_correct_macro']:.2f}%")
    print(f"    Stem correct: {macro_accuracies['by_direction_type']['stem_correct_macro']:.2f}%")
    print(f"    Direction correct: {macro_accuracies['by_direction_type']['direction_correct_macro']:.2f}%")
    mag_macro = macro_accuracies['by_direction_type']['magnitude_correct_macro']
    print(f"    Magnitude correct: {mag_macro:.2f}%" if mag_macro is not None else "    Magnitude correct: N/A")
    
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

