"""
Compare evaluation results across multiple models and plot macro accuracies.

This script loads evaluation results from multiple models and creates:
- Comparison plots of macro accuracies
- Overall performance comparisons
- Breakdown comparisons by category, stem, and direction
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_results(json_path: Path) -> dict:
    """Load evaluation results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_model_name(path: Path) -> str:
    """Extract a short model name from the path."""
    # Extract the model directory name
    parts = path.parts
    for i, part in enumerate(parts):
        if 'evaluation' in part and i + 1 < len(parts):
            model_name = parts[i + 1]
            # Shorten the name for display
            if 'all-modules' in model_name:
                return 'All Modules'
            elif 'lora-all-linear' in model_name:
                return 'LoRA All Linear'
            elif 'r16a32' in model_name and 'expanded' in model_name:
                return 'Base (Expanded)'
            else:
                return model_name.replace('qlora-qwen2-7b-mert-', '').replace('-r16a32-musdb', '')
    return path.stem


def plot_macro_accuracies_comparison(results_dict: dict, output_path: Path) -> None:
    """Plot macro accuracies comparison across models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    models = list(results_dict.keys())
    metrics = ['both_correct_macro', 'stem_correct_macro', 'direction_correct_macro']
    metric_labels = ['Both Correct', 'Stem Correct', 'Direction Correct']
    
    # Get all breakdown types
    breakdown_types = ['by_error_category', 'by_target_stem', 'by_direction_type']
    breakdown_labels = ['By Error Category', 'By Target Stem', 'By Direction Type']
    
    for idx, (breakdown_type, breakdown_label) in enumerate(zip(breakdown_types, breakdown_labels)):
        ax = axes[idx]
        
        x = np.arange(len(models))
        width = 0.25
        
        # Get values for each metric
        values = {}
        for metric, metric_label in zip(metrics, metric_labels):
            values[metric] = [
                results_dict[model]['macro_accuracies'][breakdown_type][metric]
                for model in models
            ]
        
        # Create bars
        bars1 = ax.bar(x - width, values[metrics[0]], width, label=metric_labels[0],
                      color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x, values[metrics[1]], width, label=metric_labels[1],
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
        bars3 = ax.bar(x + width, values[metrics[2]], width, label=metric_labels[2],
                      color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Macro Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Macro Accuracy: {breakdown_label}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def plot_overall_comparison(results_dict: dict, output_path: Path) -> None:
    """Plot overall performance comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = list(results_dict.keys())
    metrics = ['both_correct_percentage', 'stem_correct_percentage', 'direction_correct_percentage']
    metric_labels = ['Both Correct\n(Stem & Direction)', 'Stem Correct', 'Direction Correct']
    
    x = np.arange(len(models))
    width = 0.25
    
    values = {}
    for metric, metric_label in zip(metrics, metric_labels):
        values[metric] = [
            results_dict[model]['overall_performance'][metric]
            for model in models
        ]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars1 = ax.bar(x - width, values[metrics[0]], width, label=metric_labels[0],
                  color=colors[0], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, values[metrics[1]], width, label=metric_labels[1],
                  color=colors[1], alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + width, values[metrics[2]], width, label=metric_labels[2],
                  color=colors[2], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def plot_macro_summary(results_dict: dict, output_path: Path) -> None:
    """Plot a summary of all macro accuracies in one figure."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    models = list(results_dict.keys())
    
    # Collect all macro accuracy values
    categories = []
    values = []
    model_names = []
    
    breakdown_types = ['by_error_category', 'by_target_stem', 'by_direction_type']
    breakdown_labels = ['Error Category', 'Target Stem', 'Direction Type']
    metrics = ['both_correct_macro', 'stem_correct_macro', 'direction_correct_macro']
    metric_labels = ['Both', 'Stem', 'Direction']
    
    for model in models:
        for breakdown_type, breakdown_label in zip(breakdown_types, breakdown_labels):
            for metric, metric_label in zip(metrics, metric_labels):
                categories.append(f"{breakdown_label}\n{metric_label}")
                values.append(results_dict[model]['macro_accuracies'][breakdown_type][metric])
                model_names.append(model)
    
    # Create grouped bar chart
    x = np.arange(len(categories) // len(models))
    width = 0.25
    
    for i, model in enumerate(models):
        model_values = [values[j] for j in range(i, len(values), len(models))]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, model_values, width, label=model, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=7)
    
    # Set x-axis labels (only show unique categories)
    unique_categories = [categories[i] for i in range(0, len(categories), len(models))]
    ax.set_xticks(x)
    ax.set_xticklabels(unique_categories, rotation=45, ha='right', fontsize=9)
    
    ax.set_ylabel('Macro Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Macro Accuracy Summary Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def print_comparison_table(results_dict: dict) -> None:
    """Print a comparison table of macro accuracies."""
    print("\n" + "=" * 80)
    print("MACRO ACCURACY COMPARISON")
    print("=" * 80)
    
    models = list(results_dict.keys())
    breakdown_types = ['by_error_category', 'by_target_stem', 'by_direction_type']
    breakdown_labels = ['By Error Category', 'By Target Stem', 'By Direction Type']
    metrics = ['both_correct_macro', 'stem_correct_macro', 'direction_correct_macro']
    metric_labels = ['Both Correct', 'Stem Correct', 'Direction Correct']
    
    for breakdown_type, breakdown_label in zip(breakdown_types, breakdown_labels):
        print(f"\n{breakdown_label}:")
        print(f"{'Metric':<20}", end="")
        for model in models:
            print(f"{model:>20}", end="")
        print()
        print("-" * 80)
        
        for metric, metric_label in zip(metrics, metric_labels):
            print(f"{metric_label:<20}", end="")
            for model in models:
                value = results_dict[model]['macro_accuracies'][breakdown_type][metric]
                print(f"{value:>19.2f}%", end="")
            print()
    
    print("\n" + "=" * 80)


def main():
    """Main function to compare multiple evaluation results."""
    if len(sys.argv) < 2:
        print("Usage: python compare_evaluations.py <path1> [path2] [path3] ...")
        print("Example: python compare_evaluations.py outputs/evaluation/model1/predictions/evaluation_results.json outputs/evaluation/model2/predictions/evaluation_results.json")
        sys.exit(1)
    
    # Load all results
    results_dict = {}
    paths = []
    
    for json_path_str in sys.argv[1:]:
        json_path = Path(json_path_str)
        if not json_path.exists():
            print(f"Warning: File not found: {json_path}, skipping...")
            continue
        
        model_name = extract_model_name(json_path)
        results_dict[model_name] = load_results(json_path)
        paths.append(json_path)
    
    if not results_dict:
        print("Error: No valid evaluation results files found!")
        sys.exit(1)
    
    print(f"Loaded {len(results_dict)} evaluation results:")
    for model_name in results_dict.keys():
        print(f"  - {model_name}")
    
    # Print comparison table
    print_comparison_table(results_dict)
    
    # Create output directory for comparison plots
    # Use the parent directory of the first result file
    output_dir = paths[0].parent.parent.parent / 'comparison_plots'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating comparison plots in: {output_dir}")
    print("-" * 60)
    
    # Generate comparison plots
    plot_macro_accuracies_comparison(results_dict, output_dir / 'macro_accuracies_comparison.png')
    plot_overall_comparison(results_dict, output_dir / 'overall_comparison.png')
    plot_macro_summary(results_dict, output_dir / 'macro_summary.png')
    
    print("-" * 60)
    print(f"\n✓ All comparison plots generated successfully!")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()

