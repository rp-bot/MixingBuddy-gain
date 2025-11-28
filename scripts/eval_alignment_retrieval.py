#!/usr/bin/env python3
"""
Simple retrieval evaluation for the Audio-Text alignment model.

This script:
  - Loads the AudioTextAligner with the alignment_pretraining config
  - Loads a small subset of JSONL pairs (audio, text)
  - Computes pooled embeddings
  - Runs text→audio and audio→text retrieval using cosine similarity
  - Prints the nearest neighbours so you can qualitatively inspect alignment

Usage (from repo root):
    python scripts/eval_alignment_retrieval.py \
        --config-name alignment_pretraining \
        --checkpoint-dir outputs/checkpoints/mixing_buddy_milestone_0/alignment_pretraining_contrastive_only \
        --split val \
        --num-samples 128

Hydra will look for configs in the ../configs directory by default.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf

# Add src to path like in train_alignment
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.alignment import AudioTextAligner  # noqa: E402


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def load_jsonl(path: Path, num_samples: int) -> List[dict]:
    """Load up to num_samples entries from a JSONL file."""
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            examples.append(ex)
            if len(examples) >= num_samples:
                break
    return examples


def load_audio_batch(
    examples: List[dict],
    base_dir: Path,
    sample_rate: int,
    max_audio_length: int,
) -> torch.Tensor:
    """Load and stack audio waveforms for the given examples.

    Uses the `flawed_mix_path` field from the JSONL entries.
    """
    audio_tensors: List[torch.Tensor] = []
    for ex in examples:
        rel_path = ex.get("flawed_mix_path")
        if rel_path is None:
            continue
        audio_path = base_dir / rel_path
        if not audio_path.is_file():
            logger.warning("Missing audio file: %s", audio_path)
            continue
        wav, sr = torchaudio.load(str(audio_path))
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=sample_rate)
        # Mono mixdown if multi-channel
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)  # [time]
        # Truncate or pad to max_audio_length
        if wav.numel() > max_audio_length:
            wav = wav[:max_audio_length]
        elif wav.numel() < max_audio_length:
            pad = max_audio_length - wav.numel()
            wav = torch.nn.functional.pad(wav, (0, pad))
        audio_tensors.append(wav)

    if not audio_tensors:
        raise RuntimeError("No valid audio examples found.")

    return torch.stack(audio_tensors, dim=0)  # [B, T]


def compute_embeddings(
    model: AudioTextAligner,
    audio: torch.Tensor,
    texts: List[str],
    device: torch.device,
    use_mixed_precision: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get pooled audio and text embeddings from the model.

    We mirror the training setup by using autocast (bf16/fp16) to avoid
    dtype mismatches between Half and Float parameters.
    """
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_mixed_precision):
        batch = audio.to(device)
        outputs = model(batch, texts, return_embeddings=True)
        audio_pooled = outputs["audio_pooled"]  # [B, D]
        text_pooled = outputs["text_pooled"]    # [B, D]
    return audio_pooled, text_pooled


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity matrix between two batches of vectors."""
    a_norm = torch.nn.functional.normalize(a, p=2, dim=-1, eps=1e-8)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=-1, eps=1e-8)
    return a_norm @ b_norm.T


def truncate_to_first_sentence(texts: List[str]) -> List[str]:
    """Match training: keep only the first sentence of each text."""
    truncated: List[str] = []
    for t in texts:
        if not t:
            truncated.append("")
            continue
        s = str(t)
        idx = s.find(".")
        if idx != -1:
            s = s[: idx + 1]
        truncated.append(s.strip())
    return truncated


def run_retrieval_eval(cfg: DictConfig, args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Resolve data path based on split
    if args.split == "train":
        data_path = Path(cfg.data.train_path)
    else:
        data_path = Path(cfg.data.val_path)

    logger.info("Loading up to %d examples from %s", args.num_samples, data_path)
    examples = load_jsonl(data_path, args.num_samples)

    if not examples:
        raise RuntimeError(f"No examples loaded from {data_path}")

    # Extract and truncate texts (match training behavior)
    text_field = cfg.data.text_field
    texts_full = [ex[text_field] for ex in examples]
    texts_truncated = truncate_to_first_sentence(texts_full)

    # Load audio
    base_dir = Path(".")
    audio = load_audio_batch(
        examples,
        base_dir=base_dir,
        sample_rate=cfg.data.sample_rate,
        max_audio_length=cfg.data.max_audio_length,
    )

    # Initialize model
    logger.info("Initializing AudioTextAligner for evaluation...")
    model = AudioTextAligner(
        encoder_config=OmegaConf.to_container(cfg.model.encoder),
        qformer_config=OmegaConf.to_container(cfg.model.qformer),
        llm_model_name=cfg.model.llm_model_name,
        loss_config=OmegaConf.to_container(cfg.loss),
    )

    # Load checkpoint weights (Q-Former + MERT encoder if available)
    ckpt_dir = Path(args.checkpoint_dir)
    
    # Try new format first (model_best.pt or model_final.pt)
    model_path = ckpt_dir / "model_best.pt"
    if not model_path.is_file():
        model_path = ckpt_dir / "model_final.pt"
    
    if model_path.is_file():
        logger.info("Loading full trainable weights from %s", model_path)
        model.load_trainable_weights(str(model_path))
    else:
        # Fallback to old format
        logger.info("New model format not found. Falling back to legacy format...")
        qformer_path = ckpt_dir / "qformer_best.pt"
        if not qformer_path.is_file():
            qformer_path = ckpt_dir / "qformer_final.pt"
            
        if qformer_path.is_file():
            state_dict = torch.load(str(qformer_path), map_location="cpu")
            model.qformer.load_state_dict(state_dict)
            logger.info("Loaded Q-Former weights from %s", qformer_path)
            
            # Try to load MERT weights if they exist separately
            mert_path = ckpt_dir / "mert_encoder_best.pt"
            if not mert_path.is_file():
                mert_path = ckpt_dir / "mert_encoder_final.pt"
            
            if mert_path.is_file():
                model.load_mert_layer_weights(str(mert_path))
        else:
            logger.warning("No checkpoint found at %s", ckpt_dir)

    model.to(device)

    # Compute embeddings
    use_mixed_precision = bool(
        getattr(cfg.training, "fp16", False) or getattr(cfg.training, "bf16", False)
    )
    audio_embeds, text_embeds = compute_embeddings(
        model, audio, texts_truncated, device, use_mixed_precision=use_mixed_precision
    )

    sim_matrix = cosine_similarity_matrix(text_embeds, audio_embeds)  # [N_text, N_audio]

    # Summary stats: matched vs randomly mismatched pairs
    n = sim_matrix.shape[0]
    diag_sims = sim_matrix.diag()
    matched_mean = diag_sims.mean().item()
    matched_std = diag_sims.std(unbiased=False).item() if n > 1 else 0.0

    # Create a random permutation to form mismatched pairs (avoid identity)
    perm = torch.randperm(n)
    if n > 1:
        # Ensure no fixed points; if any index matches, rotate perm by 1
        if (perm == torch.arange(n)).any():
            perm = torch.roll(perm, shifts=1, dims=0)
        mismatched_sims = sim_matrix[torch.arange(n), perm]
        mismatched_mean = mismatched_sims.mean().item()
        mismatched_std = mismatched_sims.std(unbiased=False).item()
    else:
        mismatched_mean = matched_mean
        mismatched_std = 0.0

    logger.info(
        "Pairwise cosine stats: matched mean=%.3f (std=%.3f), mismatched mean=%.3f (std=%.3f)",
        matched_mean,
        matched_std,
        mismatched_mean,
        mismatched_std,
    )

    # Text -> audio retrieval
    logger.info("=== Text → Audio retrieval (top-3) ===")
    top_k = min(3, sim_matrix.shape[1])
    for i, ex in enumerate(examples[:10]):  # show a few queries
        sims, idxs = torch.topk(sim_matrix[i], k=top_k, dim=-1)
        # Log the truncated text actually used by the model
        logger.info("Query %d: %s", i, texts_truncated[i])
        for rank, (j, score) in enumerate(zip(idxs.tolist(), sims.tolist()), start=1):
            target = examples[j]
            logger.info(
                "  #%d (sim=%.3f) uid=%s | desc=%s",
                rank,
                score,
                target.get("global_uid", "<no_uid>"),
                target[cfg.data.text_field],
            )

    # Audio -> text retrieval (optional symmetry check)
    logger.info("=== Audio → Text retrieval (top-3) ===")
    sim_matrix_a2t = sim_matrix.T  # [N_audio, N_text]
    for i, ex in enumerate(examples[:10]):
        sims, idxs = torch.topk(sim_matrix_a2t[i], k=top_k, dim=-1)
        logger.info(
            "Audio %d: uid=%s | desc=%s",
            i,
            ex.get("global_uid", "<no_uid>"),
            texts_truncated[i],
        )
        for rank, (j, score) in enumerate(zip(idxs.tolist(), sims.tolist()), start=1):
            target = examples[j]
            logger.info(
                "  #%d (sim=%.3f) text uid=%s | desc=%s",
                rank,
                score,
                target.get("global_uid", "<no_uid>"),
                target[cfg.data.text_field],
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate alignment retrieval quality.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing qformer_final.pt or qformer_best.pt",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which JSONL split to evaluate on.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of examples to subsample for retrieval evaluation.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="alignment_pretraining",
        help="Hydra config name to use (in configs/).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point when running as a standalone script (no Hydra)."""
    args = parse_args()

    # Load config YAML directly (default: configs/alignment_pretraining.yaml)
    config_path = Path("configs") / f"{args.config_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg: DictConfig = OmegaConf.load(config_path)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    run_retrieval_eval(cfg, args)


if __name__ == "__main__":
    main()



