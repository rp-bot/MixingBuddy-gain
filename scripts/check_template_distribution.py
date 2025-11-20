#!/usr/bin/env python3
"""
Check the distribution of template usage in the variations dataset.
"""

import json
from collections import Counter
from pathlib import Path

# Load the config to get the templates
import yaml

config_path = Path("configs/data/08_musdb_expanded_augmented_variations.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

instruction_templates = config["instruction_templates"]
response_templates = config["response_templates"]

# Load the generated dataset
train_path = Path("data/musdb18hq_processed/train/training_samples_augmented_variations.jsonl")

print("Analyzing template distribution in training set...")
print(f"Total instruction templates: {len(instruction_templates)}")
print(f"Total response templates per category: {len(response_templates['no_error'])}")
print()

# Track instruction template usage
instruction_usage = Counter()
response_usage_by_category = {
    "no_error": Counter(),
    "quiet": Counter(),
    "very_quiet": Counter(),
    "loud": Counter(),
    "very_loud": Counter(),
}

total_samples = 0

with open(train_path, "r") as f:
    for line in f:
        if line.strip():
            sample = json.loads(line.strip())
            total_samples += 1
            
            # Check which instruction template was used
            # Match by trying to reconstruct the template pattern
            instruction = sample["instruction"]
            matched = False
            for idx, template in enumerate(instruction_templates):
                # Extract key unique phrases from each template
                # Each template has unique wording that distinguishes it
                template_keywords = []
                if "Listen carefully" in template:
                    template_keywords.append("Listen carefully")
                elif "Please analyze" in template:
                    template_keywords.append("Please analyze")
                elif "Examine this" in template and "carefully" in template:
                    template_keywords.append("Examine this")
                    template_keywords.append("carefully")
                elif "Review this" in template and "carefully" in template:
                    template_keywords.append("Review this")
                    template_keywords.append("carefully")
                elif "Carefully listen" in template:
                    template_keywords.append("Carefully listen")
                elif "Analyze this" in template and "Available stems:" in template:
                    template_keywords.append("Analyze this")
                    template_keywords.append("Available stems:")
                elif "Please listen" in template:
                    template_keywords.append("Please listen")
                elif "Examine this" in template and "Available stems include" in template:
                    template_keywords.append("Examine this")
                    template_keywords.append("Available stems include")
                elif "Review this" in template and "Stems available" in template:
                    template_keywords.append("Review this")
                    template_keywords.append("Stems available")
                elif "Listen to this" in template:
                    template_keywords.append("Listen to this")
                
                # Check if all keywords are present in the instruction
                if all(keyword in instruction for keyword in template_keywords):
                    instruction_usage[idx] += 1
                    matched = True
                    break
            
            if not matched:
                instruction_usage["unknown"] += 1
            
            # Check which response template was used
            response = sample["response"]
            error_category = sample["meta"]["error_category"]
            
            if error_category in response_usage_by_category:
                templates = response_templates[error_category]
                matched = False
                for idx, template in enumerate(templates):
                    # Match by unique phrases in each template
                    # Extract the unique starting phrase before the first variable
                    template_start = template.split("{")[0].strip()
                    
                    # For more accurate matching, check for unique phrases
                    # Each template has distinct wording
                    unique_phrases = []
                    if "After analyzing" in template:
                        unique_phrases.append("After analyzing")
                    elif "The mix analysis shows" in template:
                        unique_phrases.append("The mix analysis shows")
                    elif "Upon review" in template:
                        unique_phrases.append("Upon review")
                    elif "The mix is well-balanced" in template and "with all stems" in template:
                        unique_phrases.append("The mix is well-balanced")
                        unique_phrases.append("with all stems")
                    elif "After evaluation" in template:
                        unique_phrases.append("After evaluation")
                    elif "The mix analysis indicates" in template:
                        unique_phrases.append("The mix analysis indicates")
                    elif "Reviewing the mix" in template:
                        unique_phrases.append("Reviewing the mix")
                    elif "The mix is correctly balanced" in template:
                        unique_phrases.append("The mix is correctly balanced")
                    elif "After careful analysis" in template:
                        unique_phrases.append("After careful analysis")
                    elif "The mix evaluation shows" in template:
                        unique_phrases.append("The mix evaluation shows")
                    else:
                        # For error categories, match by unique phrases
                        if "a little too quiet" in template:
                            unique_phrases.append("a little too quiet")
                        elif "needs to be slightly louder" in template:
                            unique_phrases.append("needs to be slightly louder")
                        elif "somewhat quiet" in template and "Raise" in template:
                            unique_phrases.append("somewhat quiet")
                            unique_phrases.append("Raise")
                        elif "a bit too low" in template:
                            unique_phrases.append("a bit too low")
                        elif "requires a slight boost" in template:
                            unique_phrases.append("requires a slight boost")
                        elif "slightly quiet" in template and "Boost" in template and "between" in template:
                            unique_phrases.append("slightly quiet")
                            unique_phrases.append("Boost")
                            unique_phrases.append("between")
                        elif "needs more volume" in template:
                            unique_phrases.append("needs more volume")
                        elif "a little low" in template and "gain" in template:
                            unique_phrases.append("a little low")
                            unique_phrases.append("gain")
                        elif "should be slightly louder" in template:
                            unique_phrases.append("should be slightly louder")
                        elif "somewhat too quiet" in template and "Increase" in template and "by" in template:
                            unique_phrases.append("somewhat too quiet")
                            unique_phrases.append("Increase")
                            unique_phrases.append("by")
                        # Similar patterns for other error categories...
                        elif "barely audible" in template:
                            unique_phrases.append("barely audible")
                        elif "too quiet to hear clearly" in template:
                            unique_phrases.append("too quiet to hear clearly")
                        elif "nearly inaudible" in template:
                            unique_phrases.append("nearly inaudible")
                        elif "too low in the mix" in template:
                            unique_phrases.append("too low in the mix")
                        elif "difficult to hear" in template:
                            unique_phrases.append("difficult to hear")
                        elif "almost inaudible" in template:
                            unique_phrases.append("almost inaudible")
                        elif "barely present" in template:
                            unique_phrases.append("barely present")
                        elif "needs significant boosting" in template:
                            unique_phrases.append("needs significant boosting")
                        elif "too quiet to be heard properly" in template:
                            unique_phrases.append("too quiet to be heard properly")
                        elif "is too quiet" in template and "Raise" in template and "by" in template:
                            # Template 7 for very_quiet: "is too quiet" + "Raise" + "by" (not "between")
                            unique_phrases.append("is too quiet")
                            unique_phrases.append("Raise")
                            unique_phrases.append("by")
                        elif "a little too loud" in template:
                            unique_phrases.append("a little too loud")
                        elif "needs to be slightly quieter" in template:
                            unique_phrases.append("needs to be slightly quieter")
                        elif "somewhat loud" in template:
                            unique_phrases.append("somewhat loud")
                        elif "a bit too high" in template:
                            unique_phrases.append("a bit too high")
                        elif "requires a slight reduction" in template:
                            unique_phrases.append("requires a slight reduction")
                        elif "slightly loud" in template and "Reduce" in template and "between" in template:
                            unique_phrases.append("slightly loud")
                            unique_phrases.append("Reduce")
                            unique_phrases.append("between")
                        elif "needs less volume" in template:
                            unique_phrases.append("needs less volume")
                        elif "a little high" in template:
                            unique_phrases.append("a little high")
                        elif "should be slightly quieter" in template:
                            unique_phrases.append("should be slightly quieter")
                        elif "somewhat too loud" in template:
                            unique_phrases.append("somewhat too loud")
                        elif "overwhelming" in template:
                            unique_phrases.append("overwhelming")
                        elif "too loud and dominating" in template:
                            unique_phrases.append("too loud and dominating")
                        elif "overpowering the mix" in template:
                            unique_phrases.append("overpowering the mix")
                        elif "too high in the mix" in template:
                            unique_phrases.append("too high in the mix")
                        elif "excessively loud" in template:
                            unique_phrases.append("excessively loud")
                        elif "dominating the mix" in template:
                            unique_phrases.append("dominating the mix")
                        elif "overpowering" in template and "Lower" in template and "gain" in template:
                            unique_phrases.append("overpowering")
                            unique_phrases.append("Lower")
                            unique_phrases.append("gain")
                        elif "needs significant reduction" in template:
                            unique_phrases.append("needs significant reduction")
                        elif "too loud and needs to be brought down" in template:
                            unique_phrases.append("too loud and needs to be brought down")
                        elif "is too loud" in template and "Decrease" in template and "by" in template:
                            # Template 7 for very_loud: "is too loud" + "Decrease" + "by" (not "between")
                            unique_phrases.append("is too loud")
                            unique_phrases.append("Decrease")
                            unique_phrases.append("by")
                    
                    # Check if all unique phrases are in the response
                    if unique_phrases and all(phrase in response for phrase in unique_phrases):
                        response_usage_by_category[error_category][idx] += 1
                        matched = True
                        break
                
                if not matched:
                    response_usage_by_category[error_category]["unknown"] += 1

print(f"Total samples analyzed: {total_samples}")
print()
print("Instruction Template Distribution:")
print("-" * 50)
for idx in range(len(instruction_templates)):
    count = instruction_usage[idx]
    percentage = (count / total_samples) * 100
    print(f"Template {idx+1}: {count:6d} ({percentage:5.2f}%)")
if "unknown" in instruction_usage:
    print(f"Unknown: {instruction_usage['unknown']}")

print()
print("Response Template Distribution by Error Category:")
print("-" * 50)

for category in ["no_error", "quiet", "very_quiet", "loud", "very_loud"]:
    category_samples = sum(response_usage_by_category[category].values())
    if category_samples > 0:
        print(f"\n{category.upper()} ({category_samples} samples):")
        for idx in range(len(response_templates[category])):
            count = response_usage_by_category[category][idx]
            percentage = (count / category_samples) * 100 if category_samples > 0 else 0
            print(f"  Template {idx+1}: {count:6d} ({percentage:5.2f}%)")
        if "unknown" in response_usage_by_category[category]:
            print(f"  Unknown: {response_usage_by_category[category]['unknown']}")

