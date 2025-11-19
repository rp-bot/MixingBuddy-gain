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
from sklearn.metrics import confusion_matrix

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_results(json_path: Path) -> dict:
    """Load evaluation results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_predicted_error_category(generated_text: str) -> str:
    """Extract predicted error category from generated text."""
    text_lower = generated_text.lower()
    
    # Check for no_error keywords first
    no_error_keywords = ['no adjustments', 'well-balanced', 'correct level', 'balanced', 'no changes needed']
    if any(kw in text_lower for kw in no_error_keywords):
        return 'no_error'
    
    # Check for very_quiet keywords (most severe quiet)
    very_quiet_keywords = ['barely audible']
    if any(kw in text_lower for kw in very_quiet_keywords):
        return 'very_quiet'
    
    # Check for quiet keywords
    quiet_keywords = ['too quiet', 'a little too quiet']
    if any(kw in text_lower for kw in quiet_keywords):
        return 'quiet'
    
    # Check for very_loud keywords (most severe loud)
    very_loud_keywords = ['overwhelming']
    if any(kw in text_lower for kw in very_loud_keywords):
        return 'very_loud'
    
    # Check for loud keywords
    loud_keywords = ['too loud', 'a little too loud']
    if any(kw in text_lower for kw in loud_keywords):
        return 'loud'
    
    # If direction is detected but severity unclear, default based on direction
    if 'increase' in text_lower or 'too quiet' in text_lower:
        return 'quiet'  # Default to quiet if increase direction detected
    elif 'reduce' in text_lower or 'decrease' in text_lower or 'too loud' in text_lower:
        return 'loud'  # Default to loud if decrease direction detected
    
    # If nothing detected, assume no_error
    return 'no_error'


def load_predictions(jsonl_path: Path) -> list:
    """Load predictions from JSONL file."""
    predictions = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


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
                return 'projection only no lora'
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
    """Plot a summary of all macro accuracies in one figure with improved formatting."""
    fig, ax = plt.subplots(figsize=(18, 10))
    
    models = list(results_dict.keys())
    
    # Collect all macro accuracy values with better labels
    categories = []
    values = []
    model_names = []
    
    breakdown_types = ['by_error_category', 'by_target_stem', 'by_direction_type']
    breakdown_labels = ['Error Category', 'Target Stem', 'Direction Type']
    metrics = ['both_correct_macro', 'stem_correct_macro', 'direction_correct_macro']
    metric_labels = ['Both Correct\n(Stem & Direction)', 'Stem Correct\nOnly', 'Direction Correct\nOnly']
    
    for model in models:
        for breakdown_type, breakdown_label in zip(breakdown_types, breakdown_labels):
            for metric, metric_label in zip(metrics, metric_labels):
                categories.append(f"{breakdown_label}\n{metric_label}")
                values.append(results_dict[model]['macro_accuracies'][breakdown_type][metric])
                model_names.append(model)
    
    # Create grouped bar chart with better spacing
    num_categories = len(categories) // len(models)
    x = np.arange(num_categories)
    width = 0.28
    
    # Use distinct colors for each model
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, model in enumerate(models):
        model_values = [values[j] for j in range(i, len(values), len(models))]
        offset = (i - (len(models)-1)/2) * width
        bars = ax.bar(x + offset, model_values, width, label=model, 
                     color=colors[i % len(colors)], alpha=0.85, edgecolor='black', linewidth=1.5)
        
        # Add value labels with better formatting
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Set x-axis labels with better formatting
    unique_categories = [categories[i] for i in range(0, len(categories), len(models))]
    ax.set_xticks(x)
    ax.set_xticklabels(unique_categories, rotation=0, ha='center', fontsize=10, fontweight='bold')
    
    # Add vertical lines to separate breakdown types
    for i in [3, 6]:
        ax.axvline(i - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add section labels
    ax.text(1, 95, 'Error Category\nBreakdown', ha='center', fontsize=11, 
           fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(4.5, 95, 'Target Stem\nBreakdown', ha='center', fontsize=11, 
           fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(7.5, 95, 'Direction Type\nBreakdown', ha='center', fontsize=11, 
           fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel('Macro Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Metric', fontsize=14, fontweight='bold')
    
    # Main title
    title = 'Macro Accuracy Summary Comparison Across All Models'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=25)
    
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=11, title='Model', title_fontsize=12)
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.2, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def plot_all_confusion_matrices(results_dict: dict, predictions_paths: dict, output_path: Path) -> None:
    """Plot confusion matrices for all models in one figure using the same method as plot_evaluation_results.py."""
    num_models = len(results_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(10*num_models, 8))
    
    if num_models == 1:
        axes = [axes]
    
    categories = ['very_quiet', 'quiet', 'no_error', 'loud', 'very_loud']
    display_categories = [cat.replace('_', ' ').title() for cat in categories]
    
    models = list(results_dict.keys())
    
    for idx, model_name in enumerate(models):
        ax = axes[idx]
        
        # Get predictions path for this model
        if model_name not in predictions_paths:
            print(f"  ⚠ Warning: No predictions file found for {model_name}, skipping confusion matrix")
            continue
        
        predictions_path = predictions_paths[model_name]
        if not predictions_path.exists():
            print(f"  ⚠ Warning: Predictions file not found: {predictions_path}, skipping confusion matrix")
            continue
        
        # Load predictions
        predictions = load_predictions(predictions_path)
        
        # Extract ground truth and predicted categories
        y_true = []
        y_pred = []
        
        for pred in predictions:
            true_category = pred['error_category']
            generated_text = pred['generated']
            predicted_category = extract_predicted_error_category(generated_text)
            
            y_true.append(true_category)
            y_pred.append(predicted_category)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=categories)
        
        # Normalize confusion matrix to percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized) * 100  # Convert to percentages
        
        # Create heatmap (same as plot_evaluation_results.py)
        sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues', 
                    xticklabels=display_categories, yticklabels=display_categories,
                    cbar_kws={'label': 'Percentage (%)'}, ax=ax, vmin=0, vmax=100,
                    cbar=idx == num_models - 1)  # Only show colorbar on last subplot
        
        # Add raw counts as text annotations (same as plot_evaluation_results.py)
        for i in range(len(categories)):
            for j in range(len(categories)):
                if cm[i, j] > 0:
                    ax.text(j + 0.5, i + 0.7, f'({int(cm[i, j])})',
                           ha='center', va='top', fontsize=8, color='black', fontweight='bold')
        
        ax.set_xlabel('Predicted Error Category', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('True Error Category', fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel('')
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=20)
    
    fig.suptitle('Error Category Confusion Matrices Comparison', fontsize=16, fontweight='bold', y=1.02)
    
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
    predictions_paths = {}
    
    for json_path_str in sys.argv[1:]:
        json_path = Path(json_path_str)
        if not json_path.exists():
            print(f"Warning: File not found: {json_path}, skipping...")
            continue
        
        model_name = extract_model_name(json_path)
        results_dict[model_name] = load_results(json_path)
        paths.append(json_path)
        
        # Find corresponding predictions.jsonl file
        predictions_path = json_path.parent / 'predictions.jsonl'
        if predictions_path.exists():
            predictions_paths[model_name] = predictions_path
    
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
    
    # Generate confusion matrices comparison if predictions files are available
    if predictions_paths:
        plot_all_confusion_matrices(results_dict, predictions_paths, output_dir / 'confusion_matrices_comparison.png')
    
    print("-" * 60)
    print(f"\n✓ All comparison plots generated successfully!")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()

