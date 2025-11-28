#!/usr/bin/env python3
"""
Training script for Audio-Text Alignment Pre-training.

This script trains a Q-Former to align MERT audio embeddings with LLM text embeddings.
The trained Q-Former can later be used as a projection layer in the full multimodal pipeline.

Usage:
    python scripts/train_alignment.py
    python scripts/train_alignment.py training.learning_rate=5e-5 model.qformer.num_queries=64
"""

import os
import sys
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
# Note: torchaudio.set_audio_backend is deprecated in torchaudio 2.9+
# The backend is now automatically selected
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.alignment.audio_text_aligner import AudioTextAligner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AlignmentDataset(Dataset):
    """Dataset for audio-text alignment training."""
    
    def __init__(
        self,
        data_path: str,
        sample_rate: int = 32000,
        max_audio_length: int = 160000,
        text_field: str = "response",
        audio_augmentation_config: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_path: Path to JSONL file with audio paths and text
            sample_rate: Target sample rate for audio
            max_audio_length: Maximum audio length in samples
            text_field: Which field to use as text target
        """
        self.data_path = Path(data_path)
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.text_field = text_field
        self.audio_augmentation_config = audio_augmentation_config
        
        # Load data
        self.samples = []
        with open(self.data_path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append(item)
        
        if max_samples is not None and len(self.samples) > max_samples:
            # Shuffle before limiting to ensure diversity (since file is ordered by track)
            random.seed(42)  # Use fixed seed for reproducibility of the subset
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
            logger.info(f"Limited dataset to {max_samples} samples (randomly selected)")
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        
        # Get audio path - try different field names
        audio_path = item.get("flawed_mix_path") or item.get("audio_path") or item.get("mix_path")
        if audio_path is None:
            raise ValueError(f"No audio path found in item: {item.keys()}")
        
        # Load audio with robust fallback
        try:
            audio, sr = torchaudio.load(audio_path)
        except (RuntimeError, OSError, sf.LibsndfileError) as err:
            logging.warning(
                "torchaudio.load failed for %s (%s); falling back to soundfile.read",
                audio_path,
                err,
            )
            audio_np, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            audio = torch.from_numpy(audio_np)
            if audio.dim() == 2:
                audio = audio.transpose(0, 1)  # soundfile returns [samples, channels]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.squeeze(0)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Truncate or pad to max length
        if audio.shape[0] > self.max_audio_length:
            audio = audio[:self.max_audio_length]
        elif audio.shape[0] < self.max_audio_length:
            padding = self.max_audio_length - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        # Apply audio augmentations if enabled (only during training)
        if self.audio_augmentation_config and self.audio_augmentation_config.get("enabled", False):
            # Ensure audio is 2D for batch processing [1, samples]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            batch_size = 1
            device = audio.device
            
            # 1. Random gain augmentation
            gain_range_db = self.audio_augmentation_config.get("gain_range_db", 3.0)
            if gain_range_db > 0:
                random_gain_db = torch.empty(1, device=device).uniform_(-gain_range_db, gain_range_db)
                random_gain_linear = torch.pow(10.0, random_gain_db / 20.0)
                audio = audio * random_gain_linear
            
            # 2. Noise injection
            noise_snr_db_range = self.audio_augmentation_config.get("noise_snr_db_range", None)
            if noise_snr_db_range is not None:
                snr_db = torch.empty(1, device=device).uniform_(noise_snr_db_range[0], noise_snr_db_range[1])
                signal_rms = torch.sqrt(torch.mean(audio ** 2, dim=-1, keepdim=True))
                noise_rms = signal_rms / torch.pow(10.0, snr_db / 20.0)
                noise = torch.randn_like(audio)
                noise_rms_actual = torch.sqrt(torch.mean(noise ** 2, dim=-1, keepdim=True))
                noise = noise / (noise_rms_actual + 1e-8) * noise_rms
                audio = audio + noise
            
            # 3. DC offset
            dc_offset_range = self.audio_augmentation_config.get("dc_offset_range", None)
            if dc_offset_range is not None and dc_offset_range > 0:
                dc_offset = torch.empty(1, device=device).uniform_(-dc_offset_range, dc_offset_range)
                audio = audio + dc_offset.unsqueeze(-1)
                audio = torch.clamp(audio, -1.0, 1.0)
            
            # Remove batch dimension if it was added
            if audio.dim() == 2 and audio.shape[0] == 1:
                audio = audio.squeeze(0)
        
        # Get text
        if self.text_field == "both":
            text = f"{item.get('instruction', '')} {item.get('response', '')}"
        else:
            text = item.get(self.text_field, "")
        
        return {
            "audio": audio,
            "text": text,
            "global_uid": item.get("global_uid", f"sample_{idx}"),
        }


def collate_fn(batch):
    """Collate function for alignment dataset."""
    audios = torch.stack([item["audio"] for item in batch])
    texts = [item["text"] for item in batch]
    uids = [item["global_uid"] for item in batch]

    return {
        "audio": audios,
        "texts": texts,
        "global_uids": uids,
    }


def truncate_to_first_sentence(texts):
    """
    Truncate each text string to its first sentence using the first period.

    This is applied right before loss computation so that training and evaluation
    only use the first sentence of each target text.
    """
    truncated = []
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


def create_optimizer_and_scheduler(
    model: nn.Module,
    cfg: DictConfig,
    num_training_steps: int,
):
    """Create optimizer and learning rate scheduler."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )
    
    # Optimize all trainable parameters (Q-Former + Pooling layers + potentially MERT weights)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas),
    )
    
    # Warmup scheduler
    warmup_steps = int(num_training_steps * cfg.training.warmup_ratio)
    
    # Main scheduler
    if cfg.training.lr_scheduler == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=cfg.training.learning_rate * 0.01,
        )
    else:  # linear or constant
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0 if cfg.training.lr_scheduler == "linear" else 1.0,
            total_iters=num_training_steps - warmup_steps,
        )
    
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        # Combined scheduler
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = main_scheduler
    
    return optimizer, scheduler


def train_epoch(
    model: AudioTextAligner,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    cfg: DictConfig,
    epoch: int,
    global_step: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    val_loader: Optional[DataLoader] = None,
    last_eval_step: int = -1,
) -> Tuple[int, int, Optional[Dict[str, float]]]:
    """Train for one epoch."""
    model.train()
    device = next(model.qformer.parameters()).device
    
    accumulation_steps = cfg.training.gradient_accumulation_steps
    total_loss = 0.0
    total_mse_loss = 0.0
    total_cosine_loss = 0.0
    total_contrastive_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        audio = batch["audio"].to(device)
        texts = truncate_to_first_sentence(batch["texts"])

        # Log a few truncated responses for the very first training step
        if epoch == 1 and global_step == 0 and batch_idx == 0:
            sample_count = min(3, len(texts))
            logger.info("First-step truncated texts (showing up to 3 examples):")
            for i in range(sample_count):
                logger.info(f"[example {i}] {texts[i]}")
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=cfg.training.fp16 or cfg.training.bf16):
            outputs = model(audio, texts)
            loss = outputs["loss"] / accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += outputs["loss"].item()
        total_mse_loss += outputs["mse_loss"].item()
        total_cosine_loss += outputs["cosine_loss"].item()
        total_contrastive_loss += outputs["contrastive_loss"].item()
        num_batches += 1
        
        # Gradient accumulation step
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.qformer.parameters(), cfg.training.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.qformer.parameters(), cfg.training.max_grad_norm
                )
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Logging
            if global_step % cfg.training.logging_steps == 0:
                avg_loss = total_loss / num_batches
                avg_mse = total_mse_loss / num_batches
                avg_cosine = total_cosine_loss / num_batches
                avg_contrastive = total_contrastive_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                
                logger.info(
                    f"Step {global_step}: loss={avg_loss:.4f}, "
                    f"mse={avg_mse:.4f}, cosine={avg_cosine:.4f}, "
                    f"contrastive={avg_contrastive:.4f}, lr={lr:.2e}"
                )
            
            # Evaluate periodically if eval_steps is set
            eval_steps = cfg.training.get("eval_steps", None)
            val_metrics = None
            if eval_steps is not None and val_loader is not None:
                if global_step - last_eval_step >= eval_steps:
                    logger.info(f"Evaluating at step {global_step}...")
                    val_metrics = evaluate(model, val_loader, cfg)
                    logger.info(
                        f"Validation - loss={val_metrics['val_loss']:.4f}, "
                        f"mse={val_metrics['val_mse_loss']:.4f}, "
                        f"cosine={val_metrics['val_cosine_loss']:.4f}, "
                        f"contrastive={val_metrics['val_contrastive_loss']:.4f}"
                    )
                    last_eval_step = global_step
    
    return global_step, last_eval_step, val_metrics


@torch.no_grad()
def evaluate(
    model: AudioTextAligner,
    dataloader: DataLoader,
    cfg: DictConfig,
) -> Dict[str, float]:
    """Evaluate the model on validation set."""
    model.eval()
    device = next(model.qformer.parameters()).device
    
    total_loss = 0.0
    total_mse_loss = 0.0
    total_cosine_loss = 0.0
    total_contrastive_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        audio = batch["audio"].to(device)
        texts = truncate_to_first_sentence(batch["texts"])
        
        with torch.amp.autocast('cuda', enabled=cfg.training.fp16 or cfg.training.bf16):
            outputs = model(audio, texts)
        
        total_loss += outputs["loss"].item()
        total_mse_loss += outputs["mse_loss"].item()
        total_cosine_loss += outputs["cosine_loss"].item()
        total_contrastive_loss += outputs["contrastive_loss"].item()
        num_batches += 1
    
    return {
        "val_loss": total_loss / num_batches,
        "val_mse_loss": total_mse_loss / num_batches,
        "val_cosine_loss": total_cosine_loss / num_batches,
        "val_contrastive_loss": total_contrastive_loss / num_batches,
    }


def save_checkpoint(
    model: AudioTextAligner,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    global_step: int,
    output_dir: Path,
    cfg: DictConfig,
):
    """Save a training checkpoint."""
    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Q-Former weights separately (for easy loading later)
    model.save_qformer(str(checkpoint_dir / "qformer.pt"))
    
    # Save MERT encoder state (includes the 25 learnable layer combination weights)
    model.save_mert_layer_weights(str(checkpoint_dir / "mert_encoder.pt"))
    
    # Save full checkpoint for resuming
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "qformer_state_dict": model.qformer.state_dict(),
        "mert_encoder_state_dict": model.audio_encoder.state_dict() if hasattr(model.audio_encoder, "layer_weights") else None,
        "audio_pooler_state_dict": model.audio_pooler.state_dict() if hasattr(model, "audio_pooler") else None,
        "text_pooler_state_dict": model.text_pooler.state_dict() if hasattr(model, "text_pooler") else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": OmegaConf.to_container(cfg),
    }
    torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")
    
    # Save config
    OmegaConf.save(cfg, checkpoint_dir / "config.yaml")
    
    logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Cleanup old checkpoints
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if len(checkpoints) > cfg.training.save_total_limit:
        for old_ckpt in checkpoints[:-cfg.training.save_total_limit]:
            import shutil
            shutil.rmtree(old_ckpt)
            logger.info(f"Removed old checkpoint: {old_ckpt}")


@hydra.main(
    config_path="../configs",
    config_name="alignment_pretraining",
    version_base=None,
)
def main(cfg: DictConfig):
    """Main training function."""
    logger.info("=" * 60)
    logger.info("Audio-Text Alignment Pre-training")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set seed
    torch.manual_seed(cfg.env.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.env.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(cfg.env.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    logger.info("Initializing AudioTextAligner...")
    model = AudioTextAligner(
        encoder_config=OmegaConf.to_container(cfg.model.encoder),
        qformer_config=OmegaConf.to_container(cfg.model.qformer),
        llm_model_name=cfg.model.llm_model_name,
        loss_config=OmegaConf.to_container(cfg.loss),
    )
    
    # Move Q-Former to device (encoder and embed layer handle their own devices)
    model.qformer.to(device)
    model.llm_embed.to(device)
    
    model.print_trainable_parameters()
    
    # Create datasets
    logger.info("Creating datasets...")
    audio_aug_config = cfg.training.get("audio_augmentation", None)
    train_dataset = AlignmentDataset(
        data_path=cfg.data.train_path,
        sample_rate=cfg.data.sample_rate,
        max_audio_length=cfg.data.max_audio_length,
        text_field=cfg.data.text_field,
        audio_augmentation_config=OmegaConf.to_container(audio_aug_config) if audio_aug_config else None,
    )
    
    val_dataset = AlignmentDataset(
        data_path=cfg.data.val_path,
        sample_rate=cfg.data.sample_rate,
        max_audio_length=cfg.data.max_audio_length,
        text_field=cfg.data.text_field,
        audio_augmentation_config=None,  # No augmentation for validation
        max_samples=cfg.data.get("val_max_samples", None),
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
    )
    
    # Calculate training steps
    steps_per_epoch = len(train_loader) // cfg.training.gradient_accumulation_steps
    num_training_steps = steps_per_epoch * cfg.training.num_epochs
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total training steps: {num_training_steps}")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, cfg, num_training_steps)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if cfg.training.fp16 else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if cfg.resume.enabled and cfg.resume.checkpoint_path:
        checkpoint_path = Path(cfg.resume.checkpoint_path) / "checkpoint.pt"
        if checkpoint_path.exists():
            logger.info(f"Resuming from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.qformer.load_state_dict(checkpoint["qformer_state_dict"])
            if "audio_pooler_state_dict" in checkpoint and checkpoint["audio_pooler_state_dict"] is not None:
                model.audio_pooler.load_state_dict(checkpoint["audio_pooler_state_dict"])
            if "text_pooler_state_dict" in checkpoint and checkpoint["text_pooler_state_dict"] is not None:
                model.text_pooler.load_state_dict(checkpoint["text_pooler_state_dict"])
            
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
            global_step = checkpoint["global_step"]
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float("inf")
    last_eval_step = -1
    
    for epoch in range(start_epoch, cfg.training.num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
        logger.info(f"{'='*60}")
        
        # Train (with periodic evaluation if eval_steps is set)
        global_step, last_eval_step, val_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            cfg, epoch + 1, global_step, scaler,
            val_loader=val_loader, last_eval_step=last_eval_step
        )
        
        # Save checkpoint if evaluation happened and it's best
        if val_metrics is not None and val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, global_step,
                output_dir, cfg
            )
            # Also save as best
            model.save_trainable_weights(str(output_dir / "model_best.pt"))
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        
        # Always evaluate at end of epoch
        logger.info("Evaluating at end of epoch...")
        val_metrics = evaluate(model, val_loader, cfg)
        logger.info(
            f"Validation - loss={val_metrics['val_loss']:.4f}, "
            f"mse={val_metrics['val_mse_loss']:.4f}, "
            f"cosine={val_metrics['val_cosine_loss']:.4f}, "
            f"contrastive={val_metrics['val_contrastive_loss']:.4f}"
        )
        
        # Save checkpoint if best (check again in case we didn't evaluate during epoch)
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, global_step,
                output_dir, cfg
            )
            # Also save as best
            model.save_trainable_weights(str(output_dir / "model_best.pt"))
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    model.save_trainable_weights(str(output_dir / "model_final.pt"))
    logger.info(f"Training complete! Final model saved to {output_dir}")


if __name__ == "__main__":
    main()

