import json
import re

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
    
    # Check for contradictions across the entire generated text first.
    stems_to_check = ['vocals', 'drums', 'bass', '(other|other instrument)']
    for stem_pattern in stems_to_check:
        stem_sents = [s for s in generated_sents if re.search(r'\b' + stem_pattern + r'\b', s)]
        if not stem_sents:
            continue
        
        has_increase = any(any(kw in s for kw in increase_keywords) for s in stem_sents)
        has_decrease = any(any(kw in s for kw in decrease_keywords) for s in stem_sents)
        
        if has_increase and has_decrease:
            return False # Contradictory advice for a stem.

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
    stem_correct = (identified_stem == target_stem)
    direction_correct = (identified_direction == expected_direction)

    return stem_correct and direction_correct

def main():
    correct_predictions = 0
    total_predictions = 0
    filepath = 'outputs/evaluation/qlora-qwen2-7b-mert-musdb-expanded-r16a32-musdb/predictions/predictions.jsonl'
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                total_predictions += 1
                item = json.loads(line)
                if analyze_prediction(item):
                    correct_predictions += 1

    print(f"Total predictions analyzed: {total_predictions}")
    print(f"Correctly identified stem and direction: {correct_predictions}")
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()

