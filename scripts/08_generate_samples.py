"""
Qualitative evaluation script for generating and inspecting model outputs.
"""

import json
import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.generation import generate_and_compare  # noqa: E402
from src.models.checkpoint_loading import load_trained_model  # noqa: E402
from src.models.modular_multimodal_model import ModularMultimodalModel  # noqa: E402
from src.utils.model_utils import find_latest_checkpoint, load_dataset  # noqa: E402

logger = logging.getLogger(__name__)


def maybe_decode_audio_embeddings(
    cfg: DictConfig,
    model: ModularMultimodalModel,
    dataset,
    output_dir: Path,
):
    """Optionally decode projected audio embeddings to nearest text tokens."""
    decode_cfg = cfg.evaluation.get("decode_audio_embeddings")
    if not decode_cfg or not decode_cfg.get("enabled", False):
        return

    top_k = int(decode_cfg.get("top_k", 3))
    max_samples = int(decode_cfg.get("max_samples", 5))
    max_audio_tokens = decode_cfg.get("max_audio_tokens")

    if len(dataset) == 0:
        logger.warning("Dataset is empty; skipping audio embedding decoding.")
        return

    num_samples = min(max_samples, len(dataset))
    device = model.llm.device

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    output_path = predictions_dir / "audio_embedding_tokens.jsonl"

    logger.info(
        "Decoding audio embeddings to tokens for %d sample(s) (top_k=%d, max_audio_tokens=%s)",
        num_samples,
        top_k,
        str(max_audio_tokens) if max_audio_tokens is not None else "all",
    )

    with open(output_path, "w") as f:
        for idx in range(num_samples):
            sample = dataset[idx]
            audio = sample["audio"].unsqueeze(0).to(device)

            with torch.no_grad():
                audio_features = model.encode_audio(audio)
                audio_features = audio_features.to(
                    dtype=next(model.audio_projection.parameters()).dtype
                )
                projected_audio = model.audio_projection(audio_features)
                if max_audio_tokens is not None:
                    projected_audio = projected_audio[:, : int(max_audio_tokens), :]

                decoded = model.decode_audio_embeddings_to_tokens(
                    projected_audio, top_k=top_k
                )

            record = {
                "index": idx,
                "global_uid": sample.get("global_uid"),
                "target_stem": sample.get("target_stem"),
                "error_category": sample.get("error_category"),
                "instruction": sample.get("instruction"),
                "token_ids": decoded["token_ids"][0].cpu().tolist(),
                "similarities": decoded["similarities"][0].cpu().tolist(),
                "decoded_tokens": decoded["decoded_tokens"][0],
            }
            f.write(json.dumps(record) + "\n")

    logger.info("Saved audio embedding token debug file to %s", output_path)


@hydra.main(
    config_path="../configs",
    config_name="23_eval_mert_musdb_expanded_augmented_lora_all_linear_gelu",
    version_base=None,
)
def main(cfg: DictConfig):
    """
    Main generation function.
    """
    # Load model
    model: ModularMultimodalModel = load_trained_model(cfg)

    # Load test dataset
    limit = cfg.evaluation.num_generation_samples
    test_dataset = load_dataset(cfg, "test", limit=limit, random_seed=cfg.env.seed)

    # Get generation parameters from config
    max_new_tokens = cfg.evaluation.max_new_tokens
    use_instruction = cfg.data.use_instructions
    system_message = cfg.data.system_message

    # Get output directory from config - use the same run name as the checkpoint
    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path == "latest":
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            raise ValueError(
                "No checkpoint found when checkpoint_path is set to 'latest'"
            )
    checkpoint_path = Path(checkpoint_path)
    # Handle both cases: checkpoint at root or in subdirectory
    if checkpoint_path.name.startswith("checkpoint-"):
        # Checkpoint is at root level, use the parent directory name
        run_name = checkpoint_path.parent.name
        # Extract checkpoint number from checkpoint-{number}
        checkpoint_match = checkpoint_path.name.split("-")
        checkpoint_number = int(checkpoint_match[1]) if len(checkpoint_match) > 1 else None
    else:
        # Checkpoint is in a subdirectory, use the checkpoint directory name
        run_name = checkpoint_path.name
        # Try to find checkpoint number in parent directory
        checkpoint_number = None
        for parent in checkpoint_path.parents:
            if parent.name.startswith("checkpoint-"):
                checkpoint_match = parent.name.split("-")
                checkpoint_number = int(checkpoint_match[1]) if len(checkpoint_match) > 1 else None
                break

    # Create evaluation directory structure
    output_dir = Path("outputs/evaluation") / run_name

    # Get generation options from config
    use_partial_ground_truth = cfg.evaluation.get("use_partial_ground_truth", False)
    num_prefix_tokens = cfg.evaluation.get("num_prefix_tokens", 5)
    add_first_sentence_to_instruction = cfg.evaluation.get("add_first_sentence_to_instruction", False)
    use_tqdm = cfg.evaluation.get("use_tqdm", True)
    log_every_n_samples = cfg.evaluation.get("log_every_n_samples", None)

    # Generate and compare (limit can be None to generate for all samples)
    predictions_file = generate_and_compare(
        model,
        test_dataset,
        num_samples=limit,  # Can be None to generate for all samples
        max_new_tokens=max_new_tokens,
        use_instruction=use_instruction,
        system_message=system_message,
        output_dir=output_dir,
        use_partial_ground_truth=use_partial_ground_truth,
        num_prefix_tokens=num_prefix_tokens,
        checkpoint_number=checkpoint_number,
        add_first_sentence_to_instruction=add_first_sentence_to_instruction,
        use_tqdm=use_tqdm,
        log_every_n_samples=log_every_n_samples,
    )

    maybe_decode_audio_embeddings(cfg, model, test_dataset, output_dir)

    logger.info("Generation complete. Predictions saved to: %s", predictions_file)


if __name__ == "__main__":
    main()
