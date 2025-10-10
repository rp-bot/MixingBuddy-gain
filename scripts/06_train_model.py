"""
Main training script for the multimodal model.
"""

import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path
import warnings
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MixingDataset  # noqa: E402
from src.models.modular_multimodal_model import ModularMultimodalModel  # noqa: E402
from src.training.trainer import LoRATrainer  # noqa: E402
from src.utils.experiment_tracking import ExperimentTracker  # noqa: E402

# Suppress warnings
# The "resume_download" warning is a FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning)
# The "special tokens" warning is logged by the transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    """
    Main training function.
    """
    # --- 1. Initialize Model ---
    print("Initializing model...")
    model = ModularMultimodalModel(
        model_name=cfg.model.model_name,
        use_qlora=cfg.model.use_qlora,
    )
    print("Model initialized.")
    model.print_trainable_parameters()

    # --- 2. Initialize Experiment Tracker ---
    print("Initializing experiment tracker...")
    if cfg.experiment_tracking.get("use_wandb", False):
        tracker: ExperimentTracker = ExperimentTracker(
            config=cfg,
            backend="wandb",
        )
    elif cfg.experiment_tracking.get("use_mlflow", False):
        tracker: ExperimentTracker = ExperimentTracker(
            config=cfg,
            backend="mlflow",
        )
    else:
        tracker = None
    print("Experiment tracker initialized.")

    # --- 3. Load Data ---
    print("Loading data...")
    train_dataset = MixingDataset(
        jsonl_path=cfg.data.train_jsonl_path,
        audio_root=cfg.data.train_audio_root,
        tokenizer=model.tokenizer,
        sample_rate=cfg.data.audio.sample_rate,
        limit=cfg.data.get("limit"),
    )
    eval_dataset = MixingDataset(
        jsonl_path=cfg.data.eval_jsonl_path,
        audio_root=cfg.data.eval_audio_root,
        tokenizer=model.tokenizer,
        sample_rate=cfg.data.audio.sample_rate,
        limit=cfg.data.get("limit"),
    )
    x = train_dataset[0]
    print(x["audio"].shape)
    print("Data loaded.")

    # --- 4. Initialize Trainer ---
    print("Initializing trainer...")
    trainer = LoRATrainer(
        model=model,
        config=cfg,
        experiment_tracker=tracker,
    )
    print("Trainer initialized.")

    # --- 5. Train ---
    print("Starting training...")
    trainer.train(train_dataset=train_dataset, eval_dataset=eval_dataset)
    print("Training finished.")

    # --- 6. Save Model ---
    print("Saving model...")
    trainer.save_model()
    print("Model saved.")

    if tracker:
        tracker.finish()


if __name__ == "__main__":
    main()
