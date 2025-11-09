"""
Qualitative evaluation script for generating and inspecting model outputs.
"""

import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.modular_multimodal_model import ModularMultimodalModel  # noqa: E402
from src.utils.model_utils import load_dataset, find_latest_checkpoint  # noqa: E402
from src.models.checkpoint_loading import load_trained_model  # noqa: E402
from src.evaluation.generation import generate_and_compare  # noqa: E402

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs",
    config_name="16_eval_dpo",
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
    else:
        # Checkpoint is in a subdirectory, use the checkpoint directory name
        run_name = checkpoint_path.name

    # Create evaluation directory structure
    output_dir = Path("outputs/evaluation") / run_name

    # Generate and compare (limit can be None to generate for all samples)
    predictions_file = generate_and_compare(
        model,
        test_dataset,
        num_samples=limit,  # Can be None to generate for all samples
        max_new_tokens=max_new_tokens,
        use_instruction=use_instruction,
        system_message=system_message,
        output_dir=output_dir,
    )

    logger.info("Generation complete. Predictions saved to: %s", predictions_file)


if __name__ == "__main__":
    main()
