"""
Plotting utilities for evaluation metrics visualization.

This module provides clean, modular plotting functions for different types of
evaluation metrics with consistent styling and publication-quality output.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from omegaconf import DictConfig


def setup_plotting_style(cfg: DictConfig) -> None:
    """Setup plotting style from configuration."""
    # Style configuration for consistent plotting
    plot_style = {
        "figure.dpi": cfg.plotting.style.figure.dpi,
        "figure.facecolor": cfg.plotting.style.figure.facecolor,
        "axes.facecolor": "white",
        "axes.grid": cfg.plotting.style.grid.enabled,
        "font.size": cfg.plotting.style.font.size,
        "axes.titlesize": cfg.plotting.style.font.title_size,
        "axes.labelsize": cfg.plotting.style.font.label_size,
        "xtick.labelsize": cfg.plotting.style.font.tick_size,
        "ytick.labelsize": cfg.plotting.style.font.tick_size,
        "legend.fontsize": cfg.plotting.style.font.legend_size,
        "figure.titlesize": 16,
    }

    # Set global style
    plt.style.use("default")
    for key, value in plot_style.items():
        plt.rcParams[key] = value


def get_colors(cfg: DictConfig) -> Dict[str, str]:
    """Get color palette from configuration."""
    return {
        "primary": cfg.plotting.colors.primary,
        "secondary": cfg.plotting.colors.secondary,
        "success": cfg.plotting.colors.success,
        "warning": cfg.plotting.colors.warning,
        "info": cfg.plotting.colors.info,
        "light": cfg.plotting.colors.light,
        "dark": cfg.plotting.colors.dark,
        "muted": cfg.plotting.colors.muted,
        "accent": cfg.plotting.colors.accent,
    }


def setup_figure(
    figsize: Tuple[float, float] = (10, 6), title: str = "", tight_layout: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """Setup a figure with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=16, y=0.95)
    if tight_layout:
        plt.tight_layout()
    return fig, ax


def save_plot(fig: plt.Figure, output_path: Path, cfg: DictConfig) -> None:
    """Save figure in multiple formats."""
    for fmt in cfg.plotting.output.formats:
        fig.savefig(
            output_path.with_suffix(f".{fmt}"),
            format=fmt,
            dpi=cfg.plotting.output.dpi,
            bbox_inches=cfg.plotting.output.bbox_inches,
            facecolor="white",
        )


def plot_semantic_similarity_distribution(
    semantic_data: Dict, output_path: Path, cfg: DictConfig
) -> None:
    """Plot semantic similarity distribution with histogram and box plot."""
    colors = get_colors(cfg)
    per_sample = semantic_data["per_sample"]
    mean_val = semantic_data["mean"]
    median_val = semantic_data["median"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram
    ax1.hist(
        per_sample,
        bins=30,
        alpha=0.7,
        color=colors["primary"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.axvline(
        mean_val,
        color=colors["warning"],
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.3f}",
    )
    ax1.axvline(
        median_val,
        color=colors["success"],
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_val:.3f}",
    )
    ax1.set_xlabel("Semantic Similarity Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Semantic Similarity Distribution")
    ax1.legend()
    ax1.grid(
        True, alpha=cfg.plotting.style.grid.alpha, color=cfg.plotting.style.grid.color
    )

    # Box plot (horizontal)
    box_plot = ax2.boxplot(
        per_sample,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor=colors["primary"], alpha=0.7),
        medianprops=dict(color=colors["warning"], linewidth=2),
    )
    ax2.set_xlabel("Semantic Similarity Score")
    ax2.set_title("Semantic Similarity Box Plot")
    ax2.grid(
        True, alpha=cfg.plotting.style.grid.alpha, color=cfg.plotting.style.grid.color
    )

    # Add summary statistics as text
    stats_text = f"""Summary Statistics:
Mean: {mean_val:.3f}
Median: {median_val:.3f}
Std: {semantic_data["std"]:.3f}
Min: {semantic_data["min"]:.3f}
Max: {semantic_data["max"]:.3f}"""

    ax2.text(
        0.02,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.suptitle("Semantic Similarity Analysis", fontsize=16, y=0.95)
    save_plot(fig, output_path, cfg)


def plot_classification_metrics(
    metrics_data: Dict, output_path: Path, cfg: DictConfig
) -> None:
    """Plot classification metrics as bar charts."""
    colors = get_colors(cfg)
    # Extract metrics for different tasks
    error_detection = metrics_data["label_extraction"]["error_detection"]
    problem_severity = metrics_data["label_extraction"]["problem_severity"]
    direction = metrics_data["label_extraction"]["direction"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # Error Detection Metrics
    ax1 = axes[0, 0]
    error_metrics = ["accuracy", "precision", "recall", "f1"]
    error_values = [error_detection[metric] for metric in error_metrics]
    bars1 = ax1.bar(
        error_metrics,
        error_values,
        color=[
            colors["primary"],
            colors["secondary"],
            colors["success"],
            colors["warning"],
        ],
    )
    ax1.set_title("Error Detection Performance")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    ax1.grid(
        True, alpha=cfg.plotting.style.grid.alpha, color=cfg.plotting.style.grid.color
    )

    # Add value labels on bars
    for bar, value in zip(bars1, error_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    # Problem Severity Metrics
    ax2 = axes[0, 1]
    severity_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    severity_values = [problem_severity[metric] for metric in severity_metrics]
    bars2 = ax2.bar(
        severity_metrics,
        severity_values,
        color=[
            colors["primary"],
            colors["secondary"],
            colors["success"],
            colors["warning"],
        ],
    )
    ax2.set_title("Problem Severity Classification")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1)
    ax2.grid(
        True, alpha=cfg.plotting.style.grid.alpha, color=cfg.plotting.style.grid.color
    )

    for bar, value in zip(bars2, severity_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    # Direction Metrics
    ax3 = axes[1, 0]
    direction_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    direction_values = [direction[metric] for metric in direction_metrics]
    bars3 = ax3.bar(
        direction_metrics,
        direction_values,
        color=[
            colors["primary"],
            colors["secondary"],
            colors["success"],
            colors["warning"],
        ],
    )
    ax3.set_title("Direction Classification")
    ax3.set_ylabel("Score")
    ax3.set_ylim(0, 1)
    ax3.grid(
        True, alpha=cfg.plotting.style.grid.alpha, color=cfg.plotting.style.grid.color
    )

    for bar, value in zip(bars3, direction_values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    # Overall Accuracy Comparison
    ax4 = axes[1, 1]
    overall_metrics = ["Error Detection", "Problem Severity", "Direction"]
    overall_values = [
        error_detection["accuracy"],
        problem_severity["accuracy"],
        direction["accuracy"],
    ]
    bars4 = ax4.bar(
        overall_metrics,
        overall_values,
        color=[colors["primary"], colors["secondary"], colors["success"]],
    )
    ax4.set_title("Overall Accuracy Comparison")
    ax4.set_ylabel("Accuracy")
    ax4.set_ylim(0, 1)
    ax4.grid(
        True, alpha=cfg.plotting.style.grid.alpha, color=cfg.plotting.style.grid.color
    )

    for bar, value in zip(bars4, overall_values):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.suptitle("Classification Performance Metrics", fontsize=16, y=0.95)
    save_plot(fig, output_path, cfg)


def plot_per_class_metrics(
    metrics_data: Dict, output_path: Path, cfg: DictConfig
) -> None:
    """Plot per-class performance metrics."""
    colors = get_colors(cfg)
    problem_severity = metrics_data["label_extraction"]["problem_severity"]
    direction = metrics_data["label_extraction"]["direction"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    plt.subplots_adjust(wspace=0.35)

    # Problem Severity per-class metrics
    classes = list(problem_severity["precision_per_class"].keys())
    precision_vals = list(problem_severity["precision_per_class"].values())
    recall_vals = list(problem_severity["recall_per_class"].values())
    f1_vals = list(problem_severity["f1_per_class"].values())

    # Replace "unknown" with "no error" for display
    display_classes = [cls.replace("unknown", "no error") for cls in classes]

    x = np.arange(len(classes))
    width = 0.25

    bars1 = ax1.bar(
        x - width,
        precision_vals,
        width,
        label="Precision",
        color=colors["primary"],
        alpha=0.8,
    )
    bars2 = ax1.bar(
        x, recall_vals, width, label="Recall", color=colors["secondary"], alpha=0.8
    )
    bars3 = ax1.bar(
        x + width, f1_vals, width, label="F1", color=colors["success"], alpha=0.8
    )

    ax1.set_xlabel("Problem Severity Classes")
    ax1.set_ylabel("Score")
    ax1.set_title("Problem Severity Per-Class Performance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_classes, rotation=45)
    ax1.legend()
    ax1.grid(
        True, alpha=cfg.plotting.style.grid.alpha, color=cfg.plotting.style.grid.color
    )
    ax1.set_ylim(0, 1)

    # Direction per-class metrics
    dir_classes = list(direction["precision_per_class"].keys())
    dir_precision = list(direction["precision_per_class"].values())
    dir_recall = list(direction["recall_per_class"].values())
    dir_f1 = list(direction["f1_per_class"].values())

    # Replace "unknown" with "no error" for display
    display_dir_classes = [cls.replace("unknown", "no error") for cls in dir_classes]

    x2 = np.arange(len(dir_classes))

    bars4 = ax2.bar(
        x2 - width,
        dir_precision,
        width,
        label="Precision",
        color=colors["primary"],
        alpha=0.8,
    )
    bars5 = ax2.bar(
        x2, dir_recall, width, label="Recall", color=colors["secondary"], alpha=0.8
    )
    bars6 = ax2.bar(
        x2 + width, dir_f1, width, label="F1", color=colors["success"], alpha=0.8
    )

    ax2.set_xlabel("Direction Classes")
    ax2.set_ylabel("Score")
    ax2.set_title("Direction Per-Class Performance")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(display_dir_classes, rotation=45)
    ax2.legend()
    ax2.grid(
        True, alpha=cfg.plotting.style.grid.alpha, color=cfg.plotting.style.grid.color
    )
    ax2.set_ylim(0, 1)

    plt.suptitle("Per-Class Performance Metrics", fontsize=16, y=0.95)
    save_plot(fig, output_path, cfg)


def plot_confusion_matrix(
    confusion_matrix: List[List[int]], labels: List[str], title: str, ax: plt.Axes
) -> None:
    """Plot a single confusion matrix."""
    # Convert to numpy array for easier handling
    cm = np.array(confusion_matrix)

    # Normalize the confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")

    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Replace "unknown" with "no error" for display
    display_labels = [label.replace("unknown", "no error") for label in labels]

    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(display_labels, rotation=45)
    ax.set_yticklabels(display_labels)

    # Add text annotations
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


def plot_all_confusion_matrices(
    metrics_data: Dict, output_path: Path, cfg: DictConfig
) -> None:
    """Plot all confusion matrices in a single figure."""
    error_detection = metrics_data["label_extraction"]["error_detection"]
    problem_severity = metrics_data["label_extraction"]["problem_severity"]
    direction = metrics_data["label_extraction"]["direction"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    plt.subplots_adjust(wspace=0.4, left=0.05, right=0.95, top=0.9, bottom=0.15)

    # Error Detection Confusion Matrix
    plot_confusion_matrix(
        error_detection["confusion_matrix"],
        error_detection["confusion_matrix_labels"],
        "Error Detection",
        axes[0],
    )

    # Problem Severity Confusion Matrix
    plot_confusion_matrix(
        problem_severity["confusion_matrix"],
        problem_severity["confusion_matrix_labels"],
        "Problem Severity",
        axes[1],
    )

    # Direction Confusion Matrix
    plot_confusion_matrix(
        direction["confusion_matrix"],
        direction["confusion_matrix_labels"],
        "Direction",
        axes[2],
    )

    plt.suptitle("Confusion Matrices", fontsize=16, y=0.95)
    save_plot(fig, output_path, cfg)


def plot_summary_dashboard(
    metrics_data: Dict, output_path: Path, cfg: DictConfig
) -> None:
    """Create a summary dashboard with key metrics."""
    colors = get_colors(cfg)
    fig = plt.figure(figsize=(20, 12))

    # Create a grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Semantic Similarity (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    semantic_data = metrics_data["semantic_similarity"]
    per_sample = semantic_data["per_sample"]
    ax1.hist(per_sample, bins=20, alpha=0.7, color=colors["primary"])
    ax1.axvline(
        semantic_data["mean"],
        color=colors["warning"],
        linestyle="--",
        linewidth=2,
        label=f"Mean: {semantic_data['mean']:.3f}",
    )
    ax1.set_title("Semantic Similarity")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(
        True, alpha=cfg.plotting.style.grid.alpha, color=cfg.plotting.style.grid.color
    )

    # Overall Accuracy (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    label_extraction = metrics_data["label_extraction"]
    accuracy_metrics = [
        "Stem Name",
        "Magnitude Range",
        "Error Detection",
        "Problem Severity",
        "Direction",
    ]
    accuracy_values = [
        label_extraction["stem_name_accuracy"],
        label_extraction["magnitude_range_accuracy"],
        label_extraction["error_detection"]["accuracy"],
        label_extraction["problem_severity"]["accuracy"],
        label_extraction["direction"]["accuracy"],
    ]
    bars = ax2.bar(
        accuracy_metrics,
        accuracy_values,
        color=[
            colors["primary"],
            colors["secondary"],
            colors["success"],
            colors["warning"],
            colors["info"],
        ],
    )
    ax2.set_title("Accuracy by Task")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(
        True, alpha=cfg.plotting.style.grid.alpha, color=cfg.plotting.style.grid.color
    )

    # Error Detection Confusion Matrix (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_confusion_matrix(
        label_extraction["error_detection"]["confusion_matrix"],
        label_extraction["error_detection"]["confusion_matrix_labels"],
        "Error Detection",
        ax3,
    )

    # Problem Severity Confusion Matrix (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_confusion_matrix(
        label_extraction["problem_severity"]["confusion_matrix"],
        label_extraction["problem_severity"]["confusion_matrix_labels"],
        "Problem Severity",
        ax4,
    )

    # Direction Confusion Matrix (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    plot_confusion_matrix(
        label_extraction["direction"]["confusion_matrix"],
        label_extraction["direction"]["confusion_matrix_labels"],
        "Direction",
        ax5,
    )

    # F1 Scores Comparison (bottom)
    ax6 = fig.add_subplot(gs[2, :2])
    f1_metrics = ["Error Detection", "Problem Severity", "Direction"]
    f1_values = [
        label_extraction["error_detection"]["f1"],
        label_extraction["problem_severity"]["f1_macro"],
        label_extraction["direction"]["f1_macro"],
    ]
    bars = ax6.bar(
        f1_metrics,
        f1_values,
        color=[colors["primary"], colors["secondary"], colors["success"]],
    )
    ax6.set_title("F1 Scores by Task")
    ax6.set_ylabel("F1 Score")
    ax6.set_ylim(0, 1)
    ax6.grid(
        True, alpha=cfg.plotting.style.grid.alpha, color=cfg.plotting.style.grid.color
    )

    # Sample Statistics (bottom right)
    ax7 = fig.add_subplot(gs[2, 2:])
    stats_text = f"""Dataset Statistics:
Total Samples: {label_extraction["total_samples"]}
Samples with Errors: {label_extraction["samples_with_errors"]}
Overall Accuracy: {label_extraction["overall_accuracy"]:.3f}

Error Categories:
{chr(10).join([f"{k}: {v}" for k, v in label_extraction["error_category_counts"].items()])}"""

    ax7.text(
        0.05,
        0.95,
        stats_text,
        transform=ax7.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )
    ax7.set_title("Dataset Statistics")
    ax7.axis("off")

    plt.suptitle("Evaluation Metrics Dashboard", fontsize=20, y=0.95)
    save_plot(fig, output_path, cfg)
