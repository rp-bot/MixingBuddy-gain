#!/usr/bin/env python3
"""
MUSDB18 Dataset Creation Script for Automatic Mixing Research.

This script creates paired audio examples (Version A with problems, Version B balanced)
and generates corresponding text explanations for LLM fine-tuning.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse

import numpy as np
import soundfile as sf
from omegaconf import DictConfig, OmegaConf

# Add src to path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)


class MUSDBDatasetCreator:
    """Creates dataset from MUSDB18 with intentional mixing problems."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.musdb_path = Path(config.musdb.path)
        self.output_path = Path(config.output.path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # TODO: shuffle through descriptions and solutions
        # Problem templates
        self.problem_templates = {
            "vocals": {
                "buried": {
                    "gain_change": -5.0,
                    "description": "buried in the mix",
                    "solution": "increased by approximately 5dB",
                },
                "too_loud": {
                    "gain_change": 4.0,
                    "description": "too prominent in the mix",
                    "solution": "reduced by approximately 4dB",
                },
            },
            "drums": {
                "too_dominant": {
                    "gain_change": 4.0,
                    "description": "too dominant and overpowering",
                    "solution": "reduced by approximately 4dB",
                },
                "buried": {
                    "gain_change": -3.0,
                    "description": "buried and lacking punch",
                    "solution": "increased by approximately 3dB",
                },
            },
            "bass": {
                "muddy": {
                    "gain_change": 3.0,
                    "description": "muddy and overwhelming the low end",
                    "solution": "reduced by approximately 3dB",
                },
                "weak": {
                    "gain_change": -4.0,
                    "description": "weak and lacking foundation",
                    "solution": "increased by approximately 4dB",
                },
            },
            "other": {
                "harsh": {
                    "gain_change": 2.0,
                    "description": "harsh and competing with vocals",
                    "solution": "reduced by approximately 2dB",
                },
                "buried": {
                    "gain_change": -3.0,
                    "description": "buried and lacking presence",
                    "solution": "increased by approximately 3dB",
                },
            },
        }

        # Text templates
        self.text_templates = {
            "problem_description": (
                "The balance issue in Version A is with the '{stem_name}'. "
                "It is '{problem_description}'. To achieve the correct balance "
                "heard in Version B, the gain on the '{stem_name}' stem should be "
                "'{solution}'."
            ),
            "detailed_analysis": (
                "Version A has a mixing problem where the '{stem_name}' stem is "
                "'{problem_description}'. This creates an unbalanced mix. "
                "Version B demonstrates the correct balance with the '{stem_name}' "
                "stem properly leveled. The solution is to {solution} the "
                "'{stem_name}' stem to match the professional balance in Version B."
            ),
        }

    def get_musdb_tracks(self) -> List[Path]:
        """Get list of MUSDB18 track directories."""
        if not self.musdb_path.exists():
            raise FileNotFoundError(f"MUSDB18 path not found: {self.musdb_path}")

        tracks = []
        for track_dir in self.musdb_path.iterdir():
            if track_dir.is_dir():
                # Check if all required stems exist
                required_stems = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
                if all((track_dir / stem).exists() for stem in required_stems):
                    tracks.append(track_dir)

        logger.info(f"Found {len(tracks)} valid MUSDB18 tracks")
        return tracks

    def apply_gain_change(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
        """Apply gain change to audio in dB."""
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear

    def create_multi_input_audio(
        self, track_path: Path, stem_name: str, problem_type: str
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Create 4-input audio structure for Phase 2 model architecture."""

        # Load all stems
        stems = {}
        for stem_file in ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]:
            stem_path = track_path / stem_file
            audio, sample_rate = sf.read(stem_path)
            stems[stem_file.replace(".wav", "")] = audio

        # Get problem configuration
        problem_config = self.problem_templates[stem_name][problem_type]
        gain_change = problem_config["gain_change"]

        # Create Version A (problematic)
        stems_a = stems.copy()
        stems_a[stem_name] = self.apply_gain_change(stems_a[stem_name], gain_change)

        # Create Version B (balanced) - original stems
        stems_b = stems.copy()

        # Create 4-input structure
        audio_inputs = {
            # Audio 1: Problem stem from Version A
            "audio_1_problem_stem": stems_a[stem_name],
            # Audio 2: Backing tracks from Version A (all except problem stem)
            "audio_2_backing_a": sum(
                [stems_a[stem] for stem in stems_a.keys() if stem != stem_name]
            ),
            # Audio 3: Solution stem from Version B (original)
            "audio_3_solution_stem": stems_b[stem_name],
            # Audio 4: Backing tracks from Version B (all except problem stem)
            "audio_4_backing_b": sum(
                [stems_b[stem] for stem in stems_b.keys() if stem != stem_name]
            ),
        }

        # Normalize each audio input to prevent clipping
        for key, audio in audio_inputs.items():
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio_inputs[key] = audio / max_val * 0.95

        problem_info = {
            "stem_name": stem_name,
            "problem_type": problem_type,
            "gain_change_db": gain_change,
            "problem_description": problem_config["description"],
            "solution": problem_config["solution"],
        }

        return audio_inputs, problem_info

    def create_balanced_mix(self, track_path: Path) -> np.ndarray:
        """Create Version B (balanced mix) from original stems."""

        # Load and mix all stems at original levels
        mixed_audio = None
        for stem_file in ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]:
            stem_path = track_path / stem_file
            audio, _ = sf.read(stem_path)

            if mixed_audio is None:
                mixed_audio = audio
            else:
                mixed_audio += audio

        # Normalize
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 1.0:
            mixed_audio = mixed_audio / max_val * 0.95

        return mixed_audio

    def generate_text_explanation(
        self, problem_info: Dict, template_type: str = "problem_description"
    ) -> str:
        """Generate text explanation for the mixing problem."""

        template = self.text_templates[template_type]

        text = template.format(
            stem_name=problem_info["stem_name"],
            problem_description=problem_info["problem_description"],
            solution=problem_info["solution"],
        )

        return text

    def create_training_example(
        self, track_path: Path, stem_name: str, problem_type: str, output_dir: Path
    ) -> Dict:
        """Create a complete training example with 4-input audio structure."""

        track_name = track_path.name

        # Create 4-input audio structure
        audio_inputs, problem_info = self.create_multi_input_audio(
            track_path, stem_name, problem_type
        )

        # Generate text explanation using Qwen2-Audio multitask format
        instruction = f"<|startofanalysis|><|unknown|><|analysis|><|en|><|notimestamps|>Analyze the mixing balance issue in these audio inputs. You will receive 4 audio files: 1) the problematic {stem_name} stem, 2) the backing tracks from the unbalanced mix, 3) the correct {stem_name} stem, and 4) the backing tracks from the balanced mix. Identify the problem and suggest the solution."
        response = self.generate_text_explanation(problem_info)

        # Save audio files
        audio_paths = {}
        for audio_key, audio_data in audio_inputs.items():
            audio_path = (
                output_dir / f"{track_name}_{stem_name}_{problem_type}_{audio_key}.wav"
            )
            sf.write(audio_path, audio_data, 44100)
            audio_paths[audio_key] = str(audio_path)

        # Create training example optimized for Qwen2-Audio multi-input fine-tuning
        example = {
            # Core Qwen2-Audio training fields
            "instruction": instruction,
            "response": response,
            # Audio inputs for Qwen2-Audio (direct audio file paths)
            "audio_1": audio_paths["audio_1_problem_stem"],
            "audio_2": audio_paths["audio_2_backing_a"],
            "audio_3": audio_paths["audio_3_solution_stem"],
            "audio_4": audio_paths["audio_4_backing_b"],
            # Legacy format for compatibility
            "input": f"Audio 1 (Problem {stem_name}): {audio_paths['audio_1_problem_stem']}\nAudio 2 (Backing A): {audio_paths['audio_2_backing_a']}\nAudio 3 (Solution {stem_name}): {audio_paths['audio_3_solution_stem']}\nAudio 4 (Backing B): {audio_paths['audio_4_backing_b']}",
            "audio_inputs": audio_paths,
            # Metadata for analysis
            "track_name": track_name,
            "stem_name": stem_name,
            "problem_type": problem_type,
            "problem_info": problem_info,
        }

        return example

    def create_dataset(self) -> None:
        """Create the complete dataset."""

        tracks = self.get_musdb_tracks()

        # Limit tracks for testing (remove this for full dataset)
        if self.config.get("limit_tracks"):
            tracks = tracks[: self.config.limit_tracks]

        all_examples = []

        for track_path in tracks:
            logger.info(f"Processing track: {track_path.name}")

            # Create examples for each stem and problem type
            for stem_name, problems in self.problem_templates.items():
                for problem_type in problems.keys():
                    try:
                        example = self.create_training_example(
                            track_path, stem_name, problem_type, self.output_path
                        )
                        all_examples.append(example)

                    except Exception as e:
                        logger.error(
                            f"Error processing {track_path.name} - {stem_name} - {problem_type}: {e}"
                        )
                        continue

        # Save dataset metadata
        dataset_info = {
            "total_examples": len(all_examples),
            "tracks_processed": len(tracks),
            "stems": list(self.problem_templates.keys()),
            "problem_types": {
                stem: list(problems.keys())
                for stem, problems in self.problem_templates.items()
            },
        }

        # Split dataset for training/validation/test
        random.shuffle(all_examples)
        total_examples = len(all_examples)

        train_size = int(0.8 * total_examples)
        val_size = int(0.1 * total_examples)

        train_examples = all_examples[:train_size]
        val_examples = all_examples[train_size : train_size + val_size]
        test_examples = all_examples[train_size + val_size :]

        # Save splits as JSONL for LLM fine-tuning
        splits = {
            "train": train_examples,
            "validation": val_examples,
            "test": test_examples,
        }

        for split_name, examples in splits.items():
            jsonl_path = self.output_path / f"musdb_mixing_{split_name}.jsonl"
            with open(jsonl_path, "w") as f:
                for example in examples:
                    f.write(json.dumps(example) + "\n")
            logger.info(f"Saved {len(examples)} examples to {jsonl_path}")

        # Save dataset info
        dataset_info.update(
            {
                "splits": {
                    "train": len(train_examples),
                    "validation": len(val_examples),
                    "test": len(test_examples),
                }
            }
        )

        info_path = self.output_path / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)

        logger.info(f"Dataset creation complete!")
        logger.info(f"Total examples: {len(all_examples)}")
        logger.info(
            f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}"
        )
        logger.info(f"Dataset splits saved to: {self.output_path}")
        logger.info(f"Info saved to: {info_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create MUSDB18 mixing dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/musdb_dataset.yaml",
        help="Configuration file path",
    )
    parser.add_argument("--musdb-path", type=str, help="Path to MUSDB18 dataset")
    parser.add_argument("--output-path", type=str, help="Output directory path")
    parser.add_argument(
        "--limit-tracks", type=int, help="Limit number of tracks for testing"
    )

    args = parser.parse_args()

    # Load configuration
    if Path(args.config).exists():
        config = OmegaConf.load(args.config)
    else:
        # Create default config
        config = OmegaConf.create(
            {
                "musdb": {"path": args.musdb_path or "data/musdb18"},
                "output": {"path": args.output_path or "data/processed/musdb_dataset"},
                "limit_tracks": args.limit_tracks,
            }
        )

    # Override with command line args
    if args.musdb_path:
        config.musdb.path = args.musdb_path
    if args.output_path:
        config.output.path = args.output_path
    if args.limit_tracks:
        config.limit_tracks = args.limit_tracks

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create dataset
    creator = MUSDBDatasetCreator(config)
    creator.create_dataset()


if __name__ == "__main__":
    main()
