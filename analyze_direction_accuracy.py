#!/usr/bin/env python3
"""
Analyze prediction accuracy for problem detection, specifically:
1. Direction correctness (quiet vs loud)
2. Stem identification correctness
3. Magnitude range correctness
4. Exact match with ground truth
"""

import json
import re
from collections import defaultdict
from pathlib import Path

def extract_stem_and_direction(text):
    """Extract stem name and direction (quiet/loud) from first and second sentences."""
    if not text or text.strip() == "":
        return None, None
    
    # Split by sentences (period, newline, or "Assistant:")
    sentences = re.split(r'[.\n]|Assistant:', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return None, None
    
    first_sentence = sentences[0]
    second_sentence = sentences[1] if len(sentences) > 1 else ""
    
    # Check for "well-balanced" or "no adjustments"
    first_sentence_lower = first_sentence.lower()
    if "well-balanced" in first_sentence_lower or "no adjustments" in first_sentence_lower:
        return None, "no_error"
    
    # Stem names
    stems = ["vocals", "drums", "bass", "other"]
    
    # Pattern: "The [stem] is [problem description]"
    # Problem descriptions from templates:
    # - quiet: "a little too quiet"
    # - very_quiet: "barely audible"
    # - loud: "a little too loud"
    # - very_loud: "overwhelming"
    
    # Find stem in first sentence (look for "The [stem] is" pattern)
    detected_stem = None
    for stem in stems:
        # Look for pattern like "The [stem] is" or "[stem] is"
        pattern = rf"\b(the\s+)?{stem}\s+is\b"
        if re.search(pattern, first_sentence_lower):
            detected_stem = stem
            break
    
    # If not found with pattern, try simple containment as fallback
    if detected_stem is None:
        for stem in stems:
            if stem in first_sentence_lower:
                detected_stem = stem
                break
    
    # Find direction from problem description in first sentence
    direction = None
    if "barely audible" in first_sentence_lower or "too quiet" in first_sentence_lower:
        direction = "quiet"
    elif "overwhelming" in first_sentence_lower or "too loud" in first_sentence_lower:
        direction = "loud"
    
    # Also check second sentence for action (increase/reduce) to verify direction
    if second_sentence:
        second_sentence_lower = second_sentence.lower()
        # Check for action words: "increase" = quiet, "reduce" = loud
        if "increase" in second_sentence_lower:
            # If we found "increase", it should be quiet
            if direction is None:
                direction = "quiet"
            elif direction == "loud":
                # Conflict: first sentence says loud but second says increase
                # Trust the action in second sentence more
                direction = "quiet"
        elif "reduce" in second_sentence_lower:
            # If we found "reduce", it should be loud
            if direction is None:
                direction = "loud"
            elif direction == "quiet":
                # Conflict: first sentence says quiet but second says reduce
                # Trust the action in second sentence more
                direction = "loud"
    
    return detected_stem, direction

def extract_magnitude_range(text):
    """Extract magnitude range (min, max) from text, prioritizing second sentence."""
    if not text or text.strip() == "":
        return None, None
    
    # Split by sentences (period, newline, or "Assistant:")
    sentences = re.split(r'[.\n]|Assistant:', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Check second sentence first (where magnitude usually is)
    text_to_check = sentences[1] if len(sentences) > 1 else ""
    if not text_to_check:
        # Fall back to full text
        text_to_check = text
    
    text_lower = text_to_check.lower()
    
    # Pattern: "between X and Y dB" or "between Y and X dB" (can be reversed)
    # Also handle: "by X dB or more", "by X dB or less"
    
    # First check for "between X and Y dB" patterns
    between_pattern = r"between\s+([\d.]+)\s+and\s+([\d.]+)\s+db"
    match = re.search(between_pattern, text_lower)
    if match:
        val1 = float(match.group(1))
        val2 = float(match.group(2))
        return min(val1, val2), max(val1, val2)
    
    # Check for "by X dB or more/less" patterns
    by_pattern = r"by\s+([\d.]+)\s+db\s+or\s+(more|less)"
    match = re.search(by_pattern, text_lower)
    if match:
        val = float(match.group(1))
        return val, val
    
    # If not found in second sentence, check full text
    if text_to_check != text:
        text_lower = text.lower()
        match = re.search(between_pattern, text_lower)
        if match:
            val1 = float(match.group(1))
            val2 = float(match.group(2))
            return min(val1, val2), max(val1, val2)
        
        match = re.search(by_pattern, text_lower)
        if match:
            val = float(match.group(1))
            return val, val
    
    return None, None

def check_magnitude_match(pred_min, pred_max, expected_min, expected_max):
    """Check if predicted magnitude range matches expected range."""
    if expected_min is None or expected_max is None:
        # No expected range (e.g., no_error case)
        return pred_min is None and pred_max is None
    
    if pred_min is None or pred_max is None:
        return False
    
    # Check if ranges overlap or are close
    # For now, check if predicted range overlaps with expected range
    # or if predicted values are within expected range
    if pred_min <= expected_max and pred_max >= expected_min:
        return True
    
    # Also check if predicted values are within expected range
    if expected_min <= pred_min <= expected_max and expected_min <= pred_max <= expected_max:
        return True
    
    return False

def normalize_text(text):
    """Normalize text for comparison (lowercase, remove extra whitespace)."""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip().lower())
    # Remove trailing periods
    text = text.rstrip('.')
    return text

def get_ground_truth_direction(error_category):
    """Get direction from error category."""
    if error_category in ["quiet", "very_quiet"]:
        return "quiet"
    elif error_category in ["loud", "very_loud"]:
        return "loud"
    elif error_category == "no_error":
        return "no_error"
    return None

def get_expected_solution_direction(error_category):
    """Get expected solution direction (increase/reduce) from error category."""
    if error_category in ["quiet", "very_quiet"]:
        return "increase"
    elif error_category in ["loud", "very_loud"]:
        return "reduce"
    elif error_category == "no_error":
        return None  # No solution needed
    return None

def extract_solution_direction(text):
    """Extract solution direction (increase/reduce) from text, prioritizing second sentence."""
    if not text or text.strip() == "":
        return None
    
    # Split by sentences (period, newline, or "Assistant:")
    sentences = re.split(r'[.\n]|Assistant:', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Check second sentence first (where action usually is)
    text_to_check = sentences[1] if len(sentences) > 1 else ""
    if not text_to_check:
        # Fall back to full text
        text_to_check = text
    
    text_lower = text_to_check.lower()
    
    # Check for action words
    if "increase" in text_lower:
        return "increase"
    elif "reduce" in text_lower:
        return "reduce"
    
    # If not found in second sentence, check full text
    if text_to_check != text:
        text_lower = text.lower()
        if "increase" in text_lower:
            return "increase"
        elif "reduce" in text_lower:
            return "reduce"
    
    return None

def analyze_predictions(predictions_path):
    """Analyze predictions for direction and stem accuracy."""
    predictions = []
    with open(predictions_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    
    print(f"Loaded {len(predictions)} predictions\n")
    
    # Statistics
    stats = {
        "total": len(predictions),
        "correct_direction": 0,
        "correct_stem": 0,
        "correct_solution_direction": 0,
        "correct_both": 0,
        "correct_magnitude": 0,
        "correct_all_components": 0,
        "exact_match": 0,
        "direction_errors": [],
        "stem_errors": [],
        "solution_direction_errors": [],
        "magnitude_errors": [],
        "both_errors": [],
        "exact_match_errors": [],
        "by_error_category": defaultdict(lambda: {
            "total": 0,
            "correct_direction": 0,
            "correct_stem": 0,
            "correct_solution_direction": 0,
            "correct_both": 0,
            "correct_magnitude": 0,
            "correct_all_components": 0,
            "exact_match": 0,
        }),
    }
    
    for pred in predictions:
        target_stem = pred["target_stem"]
        error_category = pred["error_category"]
        ground_truth = pred["ground_truth"]
        generated = pred["generated"]
        global_uid = pred["global_uid"]
        expected_min_db = pred.get("expected_magnitude_min_db")
        expected_max_db = pred.get("expected_magnitude_max_db")
        
        # Get ground truth direction
        gt_direction = get_ground_truth_direction(error_category)
        gt_solution_direction = get_expected_solution_direction(error_category)
        
        # Extract predicted stem and direction
        pred_stem, pred_direction = extract_stem_and_direction(generated)
        
        # Extract predicted solution direction (increase/reduce)
        pred_solution_direction = extract_solution_direction(generated)
        
        # Extract predicted magnitude range
        pred_min_db, pred_max_db = extract_magnitude_range(generated)
        
        # Update category stats
        stats["by_error_category"][error_category]["total"] += 1
        
        # Check direction correctness (only for cases with errors)
        direction_correct = False
        if error_category == "no_error":
            # For no_error, prediction should also be no_error
            direction_correct = (pred_direction == "no_error" or pred_stem is None)
        else:
            # For error cases, direction should match
            direction_correct = (pred_direction == gt_direction)
        
        # Check stem correctness
        stem_correct = (pred_stem == target_stem)
        
        # Check solution direction correctness
        solution_direction_correct = False
        if error_category == "no_error":
            # For no_error, there should be no solution direction
            solution_direction_correct = (pred_solution_direction is None)
        else:
            # For error cases, solution direction should match
            solution_direction_correct = (pred_solution_direction == gt_solution_direction)
        
        # Check magnitude correctness
        magnitude_correct = check_magnitude_match(
            pred_min_db, pred_max_db, expected_min_db, expected_max_db
        )
        
        # Check exact match
        gt_normalized = normalize_text(ground_truth)
        gen_normalized = normalize_text(generated)
        exact_match = (gt_normalized == gen_normalized)
        
        # Check all components correct
        all_components_correct = (
            direction_correct and stem_correct and magnitude_correct
        )
        
        # Update stats
        if direction_correct:
            stats["correct_direction"] += 1
            stats["by_error_category"][error_category]["correct_direction"] += 1
        
        if stem_correct:
            stats["correct_stem"] += 1
            stats["by_error_category"][error_category]["correct_stem"] += 1
        
        if solution_direction_correct:
            stats["correct_solution_direction"] += 1
            stats["by_error_category"][error_category]["correct_solution_direction"] += 1
        
        if magnitude_correct:
            stats["correct_magnitude"] += 1
            stats["by_error_category"][error_category]["correct_magnitude"] += 1
        
        if direction_correct and stem_correct:
            stats["correct_both"] += 1
            stats["by_error_category"][error_category]["correct_both"] += 1
        
        if all_components_correct:
            stats["correct_all_components"] += 1
            stats["by_error_category"][error_category]["correct_all_components"] += 1
        
        if exact_match:
            stats["exact_match"] += 1
            stats["by_error_category"][error_category]["exact_match"] += 1
        
        # Record errors
        if not direction_correct:
            stats["direction_errors"].append({
                "uid": global_uid,
                "error_category": error_category,
                "gt_direction": gt_direction,
                "pred_direction": pred_direction,
                "gt_stem": target_stem,
                "pred_stem": pred_stem,
                "ground_truth": ground_truth,
                "generated": generated,
            })
        
        if not stem_correct:
            stats["stem_errors"].append({
                "uid": global_uid,
                "error_category": error_category,
                "gt_stem": target_stem,
                "pred_stem": pred_stem,
                "gt_direction": gt_direction,
                "pred_direction": pred_direction,
                "ground_truth": ground_truth,
                "generated": generated,
            })
        
        if not solution_direction_correct and error_category != "no_error":
            stats["solution_direction_errors"].append({
                "uid": global_uid,
                "error_category": error_category,
                "gt_solution_direction": gt_solution_direction,
                "pred_solution_direction": pred_solution_direction,
                "gt_direction": gt_direction,
                "pred_direction": pred_direction,
                "gt_stem": target_stem,
                "pred_stem": pred_stem,
                "ground_truth": ground_truth,
                "generated": generated,
            })
        
        if not magnitude_correct and error_category != "no_error":
            stats["magnitude_errors"].append({
                "uid": global_uid,
                "error_category": error_category,
                "gt_stem": target_stem,
                "pred_stem": pred_stem,
                "expected_min": expected_min_db,
                "expected_max": expected_max_db,
                "pred_min": pred_min_db,
                "pred_max": pred_max_db,
                "ground_truth": ground_truth,
                "generated": generated,
            })
        
        if not direction_correct and not stem_correct:
            stats["both_errors"].append({
                "uid": global_uid,
                "error_category": error_category,
                "gt_direction": gt_direction,
                "pred_direction": pred_direction,
                "gt_stem": target_stem,
                "pred_stem": pred_stem,
                "ground_truth": ground_truth,
                "generated": generated,
            })
        
        if not exact_match:
            stats["exact_match_errors"].append({
                "uid": global_uid,
                "error_category": error_category,
                "ground_truth": ground_truth,
                "generated": generated,
            })
    
    # Print results
    print("=" * 80)
    print("COMPREHENSIVE RESPONSE ACCURACY ANALYSIS")
    print("=" * 80)
    print(f"\nOverall Statistics:")
    print(f"  Total samples: {stats['total']}")
    print(f"  Correct direction (problem): {stats['correct_direction']} ({100*stats['correct_direction']/stats['total']:.1f}%)")
    print(f"  Correct stem: {stats['correct_stem']} ({100*stats['correct_stem']/stats['total']:.1f}%)")
    print(f"  Correct solution direction (increase/reduce): {stats['correct_solution_direction']} ({100*stats['correct_solution_direction']/stats['total']:.1f}%)")
    print(f"  Correct magnitude: {stats['correct_magnitude']} ({100*stats['correct_magnitude']/stats['total']:.1f}%)")
    print(f"  Correct both (direction + stem): {stats['correct_both']} ({100*stats['correct_both']/stats['total']:.1f}%)")
    print(f"  Correct all components (direction + stem + magnitude): {stats['correct_all_components']} ({100*stats['correct_all_components']/stats['total']:.1f}%)")
    print(f"  Exact match with ground truth: {stats['exact_match']} ({100*stats['exact_match']/stats['total']:.1f}%)")
    
    print(f"\nBreakdown by Error Category:")
    for category in sorted(stats["by_error_category"].keys()):
        cat_stats = stats["by_error_category"][category]
        if cat_stats["total"] > 0:
            print(f"\n  {category}:")
            print(f"    Total: {cat_stats['total']}")
            print(f"    Correct direction (problem): {cat_stats['correct_direction']} ({100*cat_stats['correct_direction']/cat_stats['total']:.1f}%)")
            print(f"    Correct stem: {cat_stats['correct_stem']} ({100*cat_stats['correct_stem']/cat_stats['total']:.1f}%)")
            print(f"    Correct solution direction (increase/reduce): {cat_stats['correct_solution_direction']} ({100*cat_stats['correct_solution_direction']/cat_stats['total']:.1f}%)")
            print(f"    Correct magnitude: {cat_stats['correct_magnitude']} ({100*cat_stats['correct_magnitude']/cat_stats['total']:.1f}%)")
            print(f"    Correct both: {cat_stats['correct_both']} ({100*cat_stats['correct_both']/cat_stats['total']:.1f}%)")
            print(f"    Correct all components: {cat_stats['correct_all_components']} ({100*cat_stats['correct_all_components']/cat_stats['total']:.1f}%)")
            print(f"    Exact match: {cat_stats['exact_match']} ({100*cat_stats['exact_match']/cat_stats['total']:.1f}%)")
    
    # print(f"\nDirection Errors: {len(stats['direction_errors'])}")
    # if stats['direction_errors']:
    #     print("\n  Sample direction errors:")
    #     for i, err in enumerate(stats['direction_errors'][:5], 1):
    #         print(f"\n    {i}. {err['uid']}")
    #         print(f"       Category: {err['error_category']}")
    #         print(f"       Expected direction: {err['gt_direction']}")
    #         print(f"       Predicted direction: {err['pred_direction']}")
    #         print(f"       Expected stem: {err['gt_stem']}")
    #         print(f"       Predicted stem: {err['pred_stem']}")
    #         # Show first sentence of generated text
    #         first_sent = err['generated'].split(".")[0].split("\n")[0].split("Assistant:")[0].strip()
    #         print(f"       First sentence (generated): {first_sent[:150]}")
    #         print(f"       Ground truth: {err['ground_truth'][:100]}...")
    
    # print(f"\nStem Errors: {len(stats['stem_errors'])}")
    # if stats['stem_errors']:
    #     print("\n  Sample stem errors:")
    #     for i, err in enumerate(stats['stem_errors'][:5], 1):
    #         print(f"\n    {i}. {err['uid']}")
    #         print(f"       Category: {err['error_category']}")
    #         print(f"       Expected stem: {err['gt_stem']}")
    #         print(f"       Predicted stem: {err['pred_stem']}")
    #         print(f"       Expected direction: {err['gt_direction']}")
    #         print(f"       Predicted direction: {err['pred_direction']}")
    #         # Show first sentence of generated text
    #         first_sent = err['generated'].split(".")[0].split("\n")[0].split("Assistant:")[0].strip()
    #         print(f"       First sentence (generated): {first_sent[:150]}")
    #         print(f"       Ground truth: {err['ground_truth'][:100]}...")
    
    # print(f"\nSolution Direction Errors: {len(stats['solution_direction_errors'])}")
    # if stats['solution_direction_errors']:
    #     print("\n  Sample solution direction errors:")
    #     for i, err in enumerate(stats['solution_direction_errors'][:5], 1):
    #         print(f"\n    {i}. {err['uid']}")
    #         print(f"       Category: {err['error_category']}")
    #         print(f"       Expected solution: {err['gt_solution_direction']}")
    #         print(f"       Predicted solution: {err['pred_solution_direction']}")
    #         print(f"       Expected problem direction: {err['gt_direction']}")
    #         print(f"       Predicted problem direction: {err['pred_direction']}")
    #         print(f"       Expected stem: {err['gt_stem']}")
    #         print(f"       Predicted stem: {err['pred_stem']}")
    #         # Show second sentence of generated text
    #         sentences = re.split(r'[.\n]|Assistant:', err['generated'])
    #         second_sent = sentences[1].strip() if len(sentences) > 1 else ""
    #         print(f"       Second sentence (generated): {second_sent[:150]}")
    #         print(f"       Ground truth: {err['ground_truth'][:100]}...")
    
    # print(f"\nMagnitude Errors: {len(stats['magnitude_errors'])}")
    # if stats['magnitude_errors']:
    #     print("\n  Sample magnitude errors:")
    #     for i, err in enumerate(stats['magnitude_errors'][:5], 1):
    #         print(f"\n    {i}. {err['uid']}")
    #         print(f"       Category: {err['error_category']}")
    #         print(f"       Expected range: {err['expected_min']} - {err['expected_max']} dB")
    #         print(f"       Predicted range: {err['pred_min']} - {err['pred_max']} dB")
    #         print(f"       Expected stem: {err['gt_stem']}")
    #         print(f"       Predicted stem: {err['pred_stem']}")
    #         # Show first sentence of generated text
    #         first_sent = err['generated'].split(".")[0].split("\n")[0].split("Assistant:")[0].strip()
    #         print(f"       First sentence (generated): {first_sent[:150]}")
    #         print(f"       Ground truth: {err['ground_truth'][:100]}...")
    
    # print(f"\nBoth Errors (wrong direction AND wrong stem): {len(stats['both_errors'])}")
    # if stats['both_errors']:
    #     print("\n  Sample errors (both wrong):")
    #     for i, err in enumerate(stats['both_errors'][:5], 1):
    #         print(f"\n    {i}. {err['uid']}")
    #         print(f"       Category: {err['error_category']}")
    #         print(f"       Expected: {err['gt_stem']} ({err['gt_direction']})")
    #         print(f"       Predicted: {err['pred_stem']} ({err['pred_direction']})")
    #         # Show first sentence of generated text
    #         first_sent = err['generated'].split(".")[0].split("\n")[0].split("Assistant:")[0].strip()
    #         print(f"       First sentence (generated): {first_sent[:150]}")
    #         print(f"       Ground truth: {err['ground_truth'][:100]}...")
    
    # print(f"\nExact Match Errors: {len(stats['exact_match_errors'])}")
    # if stats['exact_match_errors']:
    #     print("\n  Sample non-exact matches (showing differences):")
    #     for i, err in enumerate(stats['exact_match_errors'][:10], 1):
    #         print(f"\n    {i}. {err['uid']}")
    #         print(f"       Category: {err['error_category']}")
    #         print(f"       Ground truth: {err['ground_truth'][:150]}...")
    #         print(f"       Generated: {err['generated'][:150]}...")
    
    print("\n" + "=" * 80)
    
    return stats

if __name__ == "__main__":
    predictions_path = Path(
        "outputs/evaluation/qlora-qwen2-7b-mert-dpo-r8a16-musdb/predictions/predictions.jsonl"
    )
    stats = analyze_predictions(predictions_path)
    
    # Save results in the same directory as predictions
    predictions_dir = predictions_path.parent
    results_json_path = predictions_dir / "analysis_results.json"
    results_txt_path = predictions_dir / "analysis_results.txt"
    
    # Save JSON results (convert defaultdict to dict for JSON serialization)
    stats_for_json = dict(stats)
    stats_for_json["by_error_category"] = {k: dict(v) for k, v in stats["by_error_category"].items()}
    with open(results_json_path, "w") as f:
        json.dump(stats_for_json, f, indent=2)
    print(f"\nResults saved to: {results_json_path}")
    
    # Save human-readable text report
    with open(results_txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE RESPONSE ACCURACY ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nOverall Statistics:\n")
        f.write(f"  Total samples: {stats['total']}\n")
        f.write(f"  Correct direction (problem): {stats['correct_direction']} ({100*stats['correct_direction']/stats['total']:.1f}%)\n")
        f.write(f"  Correct stem: {stats['correct_stem']} ({100*stats['correct_stem']/stats['total']:.1f}%)\n")
        f.write(f"  Correct solution direction (increase/reduce): {stats['correct_solution_direction']} ({100*stats['correct_solution_direction']/stats['total']:.1f}%)\n")
        f.write(f"  Correct magnitude: {stats['correct_magnitude']} ({100*stats['correct_magnitude']/stats['total']:.1f}%)\n")
        f.write(f"  Correct both (direction + stem): {stats['correct_both']} ({100*stats['correct_both']/stats['total']:.1f}%)\n")
        f.write(f"  Correct all components (direction + stem + magnitude): {stats['correct_all_components']} ({100*stats['correct_all_components']/stats['total']:.1f}%)\n")
        f.write(f"  Exact match with ground truth: {stats['exact_match']} ({100*stats['exact_match']/stats['total']:.1f}%)\n")
        
        f.write(f"\nBreakdown by Error Category:\n")
        for category in sorted(stats["by_error_category"].keys()):
            cat_stats = stats["by_error_category"][category]
            if cat_stats["total"] > 0:
                f.write(f"\n  {category}:\n")
                f.write(f"    Total: {cat_stats['total']}\n")
                f.write(f"    Correct direction (problem): {cat_stats['correct_direction']} ({100*cat_stats['correct_direction']/cat_stats['total']:.1f}%)\n")
                f.write(f"    Correct stem: {cat_stats['correct_stem']} ({100*cat_stats['correct_stem']/cat_stats['total']:.1f}%)\n")
                f.write(f"    Correct solution direction (increase/reduce): {cat_stats['correct_solution_direction']} ({100*cat_stats['correct_solution_direction']/cat_stats['total']:.1f}%)\n")
                f.write(f"    Correct magnitude: {cat_stats['correct_magnitude']} ({100*cat_stats['correct_magnitude']/cat_stats['total']:.1f}%)\n")
                f.write(f"    Correct both: {cat_stats['correct_both']} ({100*cat_stats['correct_both']/cat_stats['total']:.1f}%)\n")
                f.write(f"    Correct all components: {cat_stats['correct_all_components']} ({100*cat_stats['correct_all_components']/cat_stats['total']:.1f}%)\n")
                f.write(f"    Exact match: {cat_stats['exact_match']} ({100*cat_stats['exact_match']/cat_stats['total']:.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    print(f"Text report saved to: {results_txt_path}")

