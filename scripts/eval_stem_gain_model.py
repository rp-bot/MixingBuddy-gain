"""
Evaluation script for stem classification and gain regression model.

Runs evaluation over the test dataset:
  data/musdb18hq_processed/test/test_samples_variations.jsonl
"""

import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path
import logging

import torch
import numpy as np
from torch.utils.data import Subset
from transformers import TrainingArguments
from transformers.trainer_utils import EvalPrediction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.stem_gain_dataset import StemGainDataset  # noqa: E402
from src.data.stem_gain_collator import StemGainDataCollator  # noqa: E402
from src.training.stem_gain_trainer import StemGainTrainer  # noqa: E402
from src.models.stem_gain_model import StemGainModel  # noqa: E402

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """
    Compute evaluation metrics for stem classification and gain regression.

    Args:
        eval_pred: EvalPrediction with:
            - predictions: tuple(classification_logits, gain_predictions)
            - label_ids: tuple(stem_labels, gain_labels)
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    # Unpack predictions and labels
    classification_logits, gain_predictions = predictions
    stem_labels, gain_labels = labels

    # Ensure numpy arrays
    classification_logits = np.asarray(classification_logits)
    gain_predictions = np.asarray(gain_predictions)
    stem_labels = np.asarray(stem_labels)
    gain_labels = np.asarray(gain_labels)

    # Stem classification accuracy
    stem_preds = classification_logits.argmax(axis=-1)
    stem_accuracy = float((stem_preds == stem_labels).mean())

    # Gain regression metrics (in dB)
    gain_errors = gain_predictions - gain_labels
    mae = float(np.mean(np.abs(gain_errors)))
    rmse = float(np.sqrt(np.mean(gain_errors ** 2)))

    return {
        "stem_accuracy": stem_accuracy,
        "gain_mae_db": mae,
        "gain_rmse_db": rmse,
    }


def load_model_from_checkpoint(cfg: DictConfig) -> StemGainModel:
    """Instantiate StemGainModel and load weights from checkpoint."""
    if cfg.checkpoint_path is None:
        raise ValueError(
            "checkpoint_path is None. Please set cfg.checkpoint_path to a trained model directory."
        )

    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    logger.info("Initializing StemGainModel for evaluation...")
    model = StemGainModel(
        encoder_config=cfg.model.get("encoder"),
        projection_config=cfg.model.get("projection"),
        num_classes=cfg.model.get("num_classes", 4),
        pooling_method=cfg.model.get("pooling_method", "mean"),
        head_config=cfg.model.get("head"),
    )

    # Load full model state dict saved by Trainer.save_model
    model_bin = checkpoint_path / "pytorch_model.bin"
    if not model_bin.exists():
        raise ValueError(f"pytorch_model.bin not found in checkpoint: {model_bin}")

    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model weights from %s", model_bin)
    state_dict = torch.load(model_bin, map_location=map_location)
    model.load_state_dict(state_dict, strict=True)

    model.to(map_location)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    logger.info("Model loaded and ready for evaluation.")
    return model


@hydra.main(config_path="../configs", config_name="evaluate_stem_gain", version_base=None)
def main(cfg: DictConfig):
    """Main evaluation entrypoint."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting stem_gain model evaluation...")
    logger.info("Checkpoint: %s", cfg.checkpoint_path)

    # Load model
    model = load_model_from_checkpoint(cfg)

    # Build test dataset (fixed paths as requested)
    test_jsonl = "data/musdb18hq_processed/test/test_samples_variations.jsonl"
    test_audio_root = "data/musdb18hq_processed/test/flawed_mixes"
    logger.info("Loading test dataset from %s", test_jsonl)

    test_dataset = StemGainDataset(
        jsonl_path=test_jsonl,
        audio_root=test_audio_root,
        sample_rate=cfg.data.audio.sample_rate,
        limit=None,
        random_seed=None,
    )

    # Optional evaluation subset
    max_samples = cfg.evaluation.max_samples
    if max_samples is not None and len(test_dataset) > max_samples:
        logger.info(
            "Limiting test set from %d to %d samples for faster evaluation",
            len(test_dataset),
            max_samples,
        )
        indices = list(range(max_samples))
        test_dataset = Subset(test_dataset, indices)

    # Data collator
    data_collator = StemGainDataCollator(pad_value=cfg.data.get("pad_value", 0.0))

    # TrainingArguments for eval-only run
    output_dir = Path(
        cfg.env.get("output_dir", "outputs/evaluation/stem_gain")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=cfg.evaluation.batch_size,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["stem_label", "gain_label"],
        logging_strategy="no",
        report_to="none",
    )

    # Trainer for evaluation
    trainer = StemGainTrainer(
        model=model,
        args=eval_args,
        train_dataset=None,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        classification_weight=cfg.training.classification_weight,
        regression_weight=cfg.training.regression_weight,
    )

    logger.info("Running evaluation on test dataset...")
    results = trainer.evaluate()
    logger.info("Evaluation complete.")
    logger.info("Results: %s", results)


if __name__ == "__main__":
    main()


