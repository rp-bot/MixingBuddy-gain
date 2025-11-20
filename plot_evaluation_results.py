"""
Plot evaluation results from evaluation_results.json files.

This script creates visualizations for the detailed evaluation metrics including:
- Overall performance metrics
- Breakdown by error category
- Breakdown by target stem
- Breakdown by direction type
"""

import json
import sys
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def plot_confusion_matrix(results_json_path: Path, predictions_jsonl_path: Path, output_path: Path) -> None:
    """Plot confusion matrix for error categories."""
    # Load predictions
    predictions = load_predictions(predictions_jsonl_path)
    
    # Extract ground truth and predicted categories
    y_true = []
    y_pred = []
    
    for pred in predictions:
        true_category = pred['error_category']
        generated_text = pred['generated']
        predicted_category = extract_predicted_error_category(generated_text)
        
        y_true.append(true_category)
        y_pred.append(predicted_category)
    
    # Define category order
    categories = ['very_quiet', 'quiet', 'no_error', 'loud', 'very_loud']
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize confusion matrix to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) * 100  # Convert to percentages
    
    # Create heatmap
    display_categories = [cat.replace('_', ' ').title() for cat in categories]
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=display_categories, yticklabels=display_categories,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax, vmin=0, vmax=100)
    
    # Add raw counts as text annotations
    for i in range(len(categories)):
        for j in range(len(categories)):
            if cm[i, j] > 0:
                ax.text(j + 0.5, i + 0.7, f'({int(cm[i, j])})',
                       ha='center', va='top', fontsize=8, color='black', fontweight='bold')
    
    ax.set_xlabel('Predicted Error Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Error Category', fontsize=12, fontweight='bold')
    ax.set_title('Error Category Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def plot_overall_performance(results: dict, output_path: Path) -> None:
    """Plot overall performance metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    overall = results['overall_performance']
    metrics = ['Both Correct\n(Stem & Direction)', 'Stem Correct']
    percentages = [
        overall['both_correct_percentage'],
        overall['stem_correct_percentage']
    ]
    
    colors = ['#2ecc71', '#3498db']
    bars = ax.bar(metrics, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add total predictions text
    total = overall['total_predictions']
    ax.text(0.02, 0.98, f'Total Predictions: {total}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def plot_by_error_category(results: dict, output_path: Path) -> None:
    """Plot performance breakdown by error category."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    by_error = results['by_error_category']
    categories = sorted(by_error.keys(), key=lambda x: (
        ['very_quiet', 'quiet', 'loud', 'very_loud', 'no_error'].index(x) 
        if x in ['very_quiet', 'quiet', 'loud', 'very_loud', 'no_error'] else 999
    ))
    
    x = np.arange(len(categories))
    width = 0.35
    
    both_vals = [by_error[cat]['both_correct_percentage'] for cat in categories]
    stem_vals = [by_error[cat]['stem_correct_percentage'] for cat in categories]
    
    # Format category names for display
    display_categories = [cat.replace('_', ' ').title() for cat in categories]
    
    bars1 = ax.bar(x - width/2, both_vals, width, label='Both Correct', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, stem_vals, width, label='Stem Correct', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Error Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Error Category', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(display_categories, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def plot_by_target_stem(results: dict, output_path: Path) -> None:
    """Plot performance breakdown by target stem."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    by_stem = results['by_target_stem']
    stems = sorted(by_stem.keys())
    
    x = np.arange(len(stems))
    width = 0.35
    
    both_vals = [by_stem[stem]['both_correct_percentage'] for stem in stems]
    stem_vals = [by_stem[stem]['stem_correct_percentage'] for stem in stems]
    
    bars1 = ax.bar(x - width/2, both_vals, width, label='Both Correct', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, stem_vals, width, label='Stem Correct', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Target Stem', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Target Stem', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([s.title() for s in stems])
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def plot_by_direction_type(results: dict, output_path: Path) -> None:
    """Plot performance breakdown by direction type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    by_direction = results['by_direction_type']
    directions = sorted(by_direction.keys(), key=lambda x: (
        ['increase', 'decrease', 'no_error'].index(x) 
        if x in ['increase', 'decrease', 'no_error'] else 999
    ))
    
    x = np.arange(len(directions))
    width = 0.35
    
    both_vals = [by_direction[dir_type]['both_correct_percentage'] for dir_type in directions]
    stem_vals = [by_direction[dir_type]['stem_correct_percentage'] for dir_type in directions]
    
    bars1 = ax.bar(x - width/2, both_vals, width, label='Both Correct', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, stem_vals, width, label='Stem Correct', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Direction Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Direction Type', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', ' ').title() if d != 'no_error' else 'No Error' for d in directions])
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def plot_summary_dashboard(results: dict, output_path: Path) -> None:
    """Create a summary dashboard with all key metrics."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Overall Performance (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    overall = results['overall_performance']
    metrics = ['Both Correct\n(Stem & Direction)', 'Stem\nCorrect']
    percentages = [
        overall['both_correct_percentage'],
        overall['stem_correct_percentage']
    ]
    colors = ['#2ecc71', '#3498db']
    bars = ax1.bar(metrics, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, value in zip(bars, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Overall Performance', fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. By Error Category (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    by_error = results['by_error_category']
    categories = sorted(by_error.keys(), key=lambda x: (
        ['very_quiet', 'quiet', 'loud', 'very_loud', 'no_error'].index(x) 
        if x in ['very_quiet', 'quiet', 'loud', 'very_loud', 'no_error'] else 999
    ))
    both_vals = [by_error[cat]['both_correct_percentage'] for cat in categories]
    display_categories = [cat.replace('_', ' ').title() for cat in categories]
    x = np.arange(len(categories))
    bars = ax2.bar(x, both_vals, color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1)
    for bar, value in zip(bars, both_vals):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_categories, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Both Correct Accuracy (%)', fontweight='bold')
    ax2.set_title('Performance by Error Category', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. By Target Stem (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    by_stem = results['by_target_stem']
    stems = sorted(by_stem.keys())
    both_vals = [by_stem[stem]['both_correct_percentage'] for stem in stems]
    x = np.arange(len(stems))
    bars = ax3.bar(x, both_vals, color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1)
    for bar, value in zip(bars, both_vals):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([s.title() for s in stems])
    ax3.set_ylabel('Both Correct Accuracy (%)', fontweight='bold')
    ax3.set_title('Performance by Target Stem', fontweight='bold', fontsize=12)
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. By Direction Type (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    by_direction = results['by_direction_type']
    directions = sorted(by_direction.keys(), key=lambda x: (
        ['increase', 'decrease', 'no_error'].index(x) 
        if x in ['increase', 'decrease', 'no_error'] else 999
    ))
    both_vals = [by_direction[dir_type]['both_correct_percentage'] for dir_type in directions]
    x = np.arange(len(directions))
    bars = ax4.bar(x, both_vals, color='#1abc9c', alpha=0.8, edgecolor='black', linewidth=1)
    for bar, value in zip(bars, both_vals):
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([d.title() for d in directions])
    ax4.set_ylabel('Both Correct Accuracy (%)', fontweight='bold')
    ax4.set_title('Performance by Direction Type', fontweight='bold', fontsize=12)
    ax4.set_ylim(0, 100)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add title
    fig.suptitle('Evaluation Results Summary Dashboard', fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def main():
    """Main function to generate all plots."""
    if len(sys.argv) < 2:
        print("Usage: python plot_evaluation_results.py <path_to_evaluation_results.json>")
        print("Example: python plot_evaluation_results.py outputs/evaluation/model/predictions/evaluation_results.json")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    print(f"Loading results from: {json_path}")
    results = load_results(json_path)
    
    # Create output directory for plots
    output_dir = json_path.parent
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating plots in: {plots_dir}")
    print("-" * 60)
    
    # Generate all plots
    plot_overall_performance(results, plots_dir / 'overall_performance.png')
    plot_by_error_category(results, plots_dir / 'by_error_category.png')
    plot_by_target_stem(results, plots_dir / 'by_target_stem.png')
    plot_by_direction_type(results, plots_dir / 'by_direction_type.png')
    plot_summary_dashboard(results, plots_dir / 'summary_dashboard.png')
    
    # Generate confusion matrix (need predictions JSONL file)
    predictions_jsonl_path = json_path.parent / 'predictions.jsonl'
    if predictions_jsonl_path.exists():
        plot_confusion_matrix(json_path, predictions_jsonl_path, plots_dir / 'confusion_matrix.png')
    else:
        print("  ⚠ Warning: Could not find predictions.jsonl file for confusion matrix")
    
    print("-" * 60)
    print(f"\n✓ All plots generated successfully!")
    print(f"  Output directory: {plots_dir}")


if __name__ == "__main__":
    main()

