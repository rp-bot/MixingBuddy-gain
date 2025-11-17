import logging
from typing import Tuple

from omegaconf import DictConfig

from src.data.dataset import MixingDataset
from src.utils.model_utils import IterableDatasetWrapper


logger = logging.getLogger(__name__)


def load_datasets(cfg: DictConfig, tokenizer) -> Tuple[object, object, object]:
    """Load and split train, validation, and test datasets."""
    logger.info(
        "Expected audio length: %d samples (%ss at %sHz)",
        int(cfg.data.chunk.sec * cfg.data.audio.sample_rate),
        cfg.data.chunk.sec,
        cfg.data.audio.sample_rate,
    )

    full_train_pytorch_dataset = MixingDataset(
        jsonl_path=cfg.data.train_jsonl_path,
        audio_root=cfg.data.train_audio_root,
        sample_rate=cfg.data.audio.sample_rate,
        limit=cfg.data.limit,
        use_instructions=cfg.data.use_instructions,
        system_message=cfg.data.system_message,
    )
    full_train_dataset = IterableDatasetWrapper(full_train_pytorch_dataset)

    test_pytorch_dataset = MixingDataset(
        jsonl_path=cfg.data.test_jsonl_path,
        audio_root=cfg.data.test_audio_root,
        sample_rate=cfg.data.audio.sample_rate,
        limit=cfg.data.limit,
        use_instructions=cfg.data.use_instructions,
        system_message=cfg.data.system_message,
    )
    test_dataset = IterableDatasetWrapper(test_pytorch_dataset)

    train_val_split = full_train_dataset.train_test_split(
        test_size=0.2, seed=cfg.env.seed
    )
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    # Optionally limit validation set size for faster evaluation
    max_eval_samples = cfg.training.get("evaluation", {}).get("max_eval_samples", None)
    if max_eval_samples is not None and len(val_dataset) > max_eval_samples:
        logger.info(
            "Limiting validation set from %d to %d samples for faster evaluation",
            len(val_dataset),
            max_eval_samples,
        )
        # Select first max_eval_samples (dataset is already shuffled by train_test_split)
        val_dataset = val_dataset.select(range(max_eval_samples))

    logger.info(
        "Dataset sizes: Train=%d, Val=%d, Test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )
    return train_dataset, val_dataset, test_dataset
