#!/usr/bin/env python3
"""
Visualize dataset samples to understand what the model will see.
Shows waveforms, spectrograms, and metadata for debugging.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import soundfile as sf
from transformers import AutoTokenizer

from src.data.dataset import MixingDataset
from src.utils.audio_utils import load_audio_chunk, to_mono


def plot_audio_sample(sample, sample_idx, output_dir):
    """Create a comprehensive visualization of a single dataset sample."""

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Get audio data (separate fields)
    anchor_audio = sample["anchor_audio"].numpy()
    mix_audio = sample["mix_audio"].numpy()
    silence_samples = int(sample.get("silence_samples", 0))
    sample_rate = int(sample.get("sample_rate", 48000))

    # Build concatenated waveform for full-plot view
    if silence_samples > 0:
        silence = np.zeros(silence_samples, dtype=np.float32)
        audio_full = np.concatenate([anchor_audio, silence, mix_audio])
    else:
        audio_full = np.concatenate([anchor_audio, mix_audio])

    # Calculate time axis
    duration = len(audio_full) / sample_rate
    time_axis = np.linspace(0, duration, len(audio_full))

    # Durations derived from actual arrays
    anchor_duration = len(anchor_audio) / sample_rate
    silence_duration = silence_samples / sample_rate
    mix_duration = len(mix_audio) / sample_rate

    # Plot 1: Full waveform
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, audio_full, "b-", linewidth=0.5, alpha=0.7)
    ax1.set_title(
        f"Sample {sample_idx}: Full Audio Waveform (Anchor + Silence + Mix)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    # Add region labels
    ax1.axvline(
        x=anchor_duration, color="r", linestyle="--", alpha=0.7, label="Anchor end"
    )
    ax1.axvline(
        x=anchor_duration + silence_duration,
        color="g",
        linestyle="--",
        alpha=0.7,
        label="Mix start",
    )

    # Add colored regions
    ax1.axvspan(0, anchor_duration, alpha=0.1, color="blue", label="Anchor")
    ax1.axvspan(
        anchor_duration,
        anchor_duration + silence_duration,
        alpha=0.1,
        color="gray",
        label="Silence",
    )
    ax1.axvspan(
        anchor_duration + silence_duration,
        duration,
        alpha=0.1,
        color="red",
        label="Flawed Mix",
    )

    ax1.legend()
    ax1.set_xlim(0, duration)

    # Plot 2: Anchor waveform (detailed)
    ax2 = fig.add_subplot(gs[1, 0])
    anchor_time = np.linspace(0, anchor_duration, len(anchor_audio))
    ax2.plot(anchor_time, anchor_audio, "b-", linewidth=0.8)
    ax2.set_title(f"Anchor: {sample['anchor_stem']}", fontweight="bold")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, anchor_duration)

    # Plot 3: Flawed mix waveform (detailed)
    ax3 = fig.add_subplot(gs[1, 1])
    mix_time = np.linspace(0, mix_duration, len(mix_audio))
    ax3.plot(mix_time, mix_audio, "r-", linewidth=0.8)
    ax3.set_title(f"Flawed Mix (Error: {sample['error_category']})", fontweight="bold")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Amplitude")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, mix_duration)

    # Plot 4: Spectrogram of anchor
    ax4 = fig.add_subplot(gs[2, 0])
    try:
        ax4.specgram(
            anchor_audio, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="viridis"
        )
        ax4.set_title("Anchor Spectrogram", fontweight="bold")
        ax4.set_xlabel("Time (seconds)")
        ax4.set_ylabel("Frequency (Hz)")
    except Exception as e:
        ax4.text(
            0.5,
            0.5,
            f"Spectrogram error: {str(e)[:50]}...",
            transform=ax4.transAxes,
            ha="center",
            va="center",
        )
        ax4.set_title("Anchor Spectrogram (Error)", fontweight="bold")

    # Plot 5: Spectrogram of flawed mix
    ax5 = fig.add_subplot(gs[2, 1])
    try:
        ax5.specgram(mix_audio, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="viridis")
        ax5.set_title("Flawed Mix Spectrogram", fontweight="bold")
        ax5.set_xlabel("Time (seconds)")
        ax5.set_ylabel("Frequency (Hz)")
    except Exception as e:
        ax5.text(
            0.5,
            0.5,
            f"Spectrogram error: {str(e)[:50]}...",
            transform=ax5.transAxes,
            ha="center",
            va="center",
        )
        ax5.set_title("Flawed Mix Spectrogram (Error)", fontweight="bold")

    # Plot 6: RMS levels comparison
    ax6 = fig.add_subplot(gs[3, :])

    # Calculate RMS in windows
    window_size = int(0.1 * sample_rate)  # 100ms windows
    anchor_rms = []
    mix_rms = []

    for i in range(0, len(anchor_audio) - window_size, window_size):
        anchor_rms.append(np.sqrt(np.mean(anchor_audio[i : i + window_size] ** 2)))

    for i in range(0, len(mix_audio) - window_size, window_size):
        mix_rms.append(np.sqrt(np.mean(mix_audio[i : i + window_size] ** 2)))

    anchor_rms_time = np.linspace(0, anchor_duration, len(anchor_rms))
    mix_rms_time = np.linspace(
        anchor_duration + silence_duration, duration, len(mix_rms)
    )

    ax6.plot(anchor_rms_time, anchor_rms, "b-", linewidth=2, label="Anchor RMS")
    ax6.plot(mix_rms_time, mix_rms, "r-", linewidth=2, label="Mix RMS")
    ax6.set_title("RMS Level Comparison (100ms windows)", fontweight="bold")
    ax6.set_xlabel("Time (seconds)")
    ax6.set_ylabel("RMS Level")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, duration)

    # Add metadata text
    metadata_text = f"""
Sample Info:
• Global UID: {sample["global_uid"]}
• Anchor Stem: {sample["anchor_stem"]}
• Target Stem: {sample["target_stem"]}
• Error Category: {sample["error_category"]}
• Instruction: {sample["instruction"][:100]}...
• Response: {sample["response"][:100]}...
• Audio Length: {len(audio_full)} samples ({duration:.2f}s)
• Sample Rate: {sample_rate} Hz
    """

    fig.text(
        0.02,
        0.02,
        metadata_text,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    # Save plot
    output_path = output_dir / f"dataset_visualization_sample_{sample_idx}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved visualization: {output_path}")
    return output_path


def analyze_dataset_stats(dataset, num_samples=10):
    """Analyze basic statistics of the dataset."""
    print(f"\n📊 Dataset Analysis (first {num_samples} samples):")
    print("=" * 60)

    audio_lengths = []
    error_categories = {}
    anchor_stems = {}
    target_stems = {}

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        # Audio length
        audio_lengths.append(len(sample["audio"]))

        # Error categories
        error_cat = sample["error_category"]
        error_categories[error_cat] = error_categories.get(error_cat, 0) + 1

        # Anchor stems
        anchor = sample["anchor_stem"]
        anchor_stems[anchor] = anchor_stems.get(anchor, 0) + 1

        # Target stems
        target = sample["target_stem"]
        target_stems[target] = target_stems.get(target, 0) + 1

    # Print statistics
    print(f"Audio Lengths:")
    print(
        f"  Mean: {np.mean(audio_lengths):.0f} samples ({np.mean(audio_lengths) / 48000:.2f}s)"
    )
    print(
        f"  Min: {np.min(audio_lengths):.0f} samples ({np.min(audio_lengths) / 48000:.2f}s)"
    )
    print(
        f"  Max: {np.max(audio_lengths):.0f} samples ({np.max(audio_lengths) / 48000:.2f}s)"
    )

    print(f"\nError Categories:")
    for cat, count in sorted(error_categories.items()):
        print(f"  {cat}: {count}")

    print(f"\nAnchor Stems:")
    for stem, count in sorted(anchor_stems.items()):
        print(f"  {stem}: {count}")

    print(f"\nTarget Stems:")
    for stem, count in sorted(target_stems.items()):
        print(f"  {stem}: {count}")


def main():
    """Main visualization function."""
    print("🎨 Dataset Visualization Tool")
    print("=" * 40)

    # Paths
    jsonl_path = "data/musdb18hq_processed/train/training_samples.jsonl"
    audio_root = "data"
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Check if files exist
    if not Path(jsonl_path).exists():
        print(f"❌ JSONL file not found: {jsonl_path}")
        return

    try:
        # Load tokenizer
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded")

        # Create dataset
        print("📊 Creating dataset...")
        dataset = MixingDataset(
            jsonl_path=jsonl_path,
            audio_root=audio_root,
            tokenizer=tokenizer,
            limit=10,  # Just visualize first 10 samples
        )
        print(f"✅ Dataset created with {len(dataset)} samples")

        # Analyze dataset statistics
        analyze_dataset_stats(dataset, num_samples=len(dataset))

        # Visualize samples
        print(f"\n🎨 Creating visualizations...")
        for i in range(min(3, len(dataset))):  # Visualize first 3 samples
            print(f"  Processing sample {i}...")
            sample = dataset[i]
            plot_audio_sample(sample, i, output_dir)

        print(f"\n🎉 Visualization complete!")
        print(f"📁 Check the 'outputs/' directory for visualization images")

    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
