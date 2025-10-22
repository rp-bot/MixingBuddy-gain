"""
Generate visualization plots for evaluation metrics.

This script loads metrics from a JSON file and generates clean, publication-quality
plots for semantic similarity, classification performance, confusion matrices,
and a comprehensive dashboard.
"""

import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path
import json
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.plotting import (
    setup_plotting_style,
    plot_semantic_similarity_distribution,
    plot_classification_metrics,
    plot_per_class_metrics,
    plot_all_confusion_matrices,
    plot_summary_dashboard,
)


def load_metrics(metrics_path: Path) -> Dict:
    """Load metrics from JSON file."""
    print(f"Loading metrics from {metrics_path}")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    print(f"Loaded metrics with {len(metrics)} top-level keys")
    return metrics


def create_output_directory(output_path: Path) -> Path:
    """Create output directory for plots."""
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {plots_dir}")
    return plots_dir


def generate_all_plots(metrics: Dict, output_dir: Path, cfg: DictConfig) -> None:
    """Generate all visualization plots."""
    print("Generating visualization plots...")

    # Setup plotting style
    setup_plotting_style(cfg)

    # 1. Semantic Similarity Distribution
    if cfg.plotting.generate.semantic_similarity:
        print("  - Generating semantic similarity distribution plot...")
        plot_semantic_similarity_distribution(
            metrics["semantic_similarity"], output_dir / "01_semantic_similarity", cfg
        )

    # 2. Classification Metrics
    if cfg.plotting.generate.classification_metrics:
        print("  - Generating classification metrics plot...")
        plot_classification_metrics(
            metrics, output_dir / "02_classification_metrics", cfg
        )

    # 3. Per-Class Performance
    if cfg.plotting.generate.per_class_metrics:
        print("  - Generating per-class metrics plot...")
        plot_per_class_metrics(metrics, output_dir / "03_per_class_metrics", cfg)

    # 4. Confusion Matrices
    if cfg.plotting.generate.confusion_matrices:
        print("  - Generating confusion matrices plot...")
        plot_all_confusion_matrices(metrics, output_dir / "04_confusion_matrices", cfg)

    # 5. Summary Dashboard
    if cfg.plotting.generate.summary_dashboard:
        print("  - Generating summary dashboard...")
        plot_summary_dashboard(metrics, output_dir / "05_summary_dashboard", cfg)

    print("All plots generated successfully!")


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    """Main function to orchestrate plot generation."""
    # Determine output directory from checkpoint path (same logic as evaluation)
    checkpoint_path = Path(cfg.checkpoint_path)
    if checkpoint_path.name.startswith("checkpoint-"):
        # Extract run name from checkpoint path
        # e.g., "outputs/checkpoints/mixing_buddy_milestone_0/qlora-qwen2-7b-all-linear-r8a16-musdb-1020-2051/checkpoint-500"
        # -> "qlora-qwen2-7b-all-linear-r8a16-musdb-1020-2051"
        run_name = checkpoint_path.parent.name
    else:
        # If not a checkpoint path, use the directory name
        run_name = checkpoint_path.name

    # Set paths based on run name
    output_dir = PROJECT_ROOT / "outputs" / "evaluation" / run_name / "predictions"
    metrics_file = output_dir / "metrics_results_detailed.json"

    # Override config with determined paths
    cfg.plotting.metrics_file = str(metrics_file)
    cfg.plotting.output_dir = str(output_dir)

    print(f"Using metrics file: {metrics_file}")
    print(f"Using output directory: {output_dir}")

    # Validate inputs
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    # Load metrics
    metrics = load_metrics(metrics_file)

    # Create output directory
    plots_dir = create_output_directory(output_dir)

    # Generate all plots
    generate_all_plots(metrics, plots_dir, cfg)

    print(f"\nAll plots saved to: {plots_dir}")
    print("Generated files:")
    for plot_file in sorted(plots_dir.glob("*.png")):
        print(f"  - {plot_file.name}")


if __name__ == "__main__":
    main()
