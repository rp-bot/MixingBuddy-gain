
import torch
import hydra
from omegaconf import OmegaConf
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, os.getcwd())

from src.models.alignment.audio_text_aligner import AudioTextAligner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    # Create a dummy config
    cfg = OmegaConf.create({
        "model": {
            "encoder": {
                "model_name": "m-a-p/MERT-v1-330M",
                "freeze": True,
                "input_sample_rate": 24000,
                "crop_audio": False
            },
            "qformer": {
                "num_queries": 16,
                "num_layers": 4, # Match training config
                "num_heads": 8,  # Match training config
                "hidden_dim": 512,
                "dropout": 0.1,
                "feedforward_dim": 2048
            },
            "llm_model_name": "Qwen/Qwen2-7B-Instruct",
            "pooling_type": "attention"
        },
        "loss": {
            "mse_weight": 0.0,
            "cosine_weight": 0.0,
            "contrastive_weight": 1.0,
            "temperature": 0.07
        }
    })

    logger.info("Initializing model...")
    try:
        model = AudioTextAligner(
            encoder_config=OmegaConf.to_container(cfg.model.encoder),
            qformer_config=OmegaConf.to_container(cfg.model.qformer),
            llm_model_name=cfg.model.llm_model_name,
            loss_config=OmegaConf.to_container(cfg.loss),
            pooling_type=cfg.model.pooling_type
        )
    except Exception as e:
        logger.error(f"Failed to init model: {e}")
        return

    # Load latest checkpoint
    checkpoint_dir = Path("outputs/checkpoints/mixing_buddy_milestone_0/alignment_pretraining_contrastive_only")
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if checkpoints:
        latest_ckpt = checkpoints[-1]
        logger.info(f"Loading weights from {latest_ckpt}...")
        # Load Q-Former
        qformer_path = latest_ckpt / "qformer.pt"
        if qformer_path.exists():
            model.load_qformer(str(qformer_path))
        else:
            logger.warning(f"No qformer.pt found in {latest_ckpt}")
            
        # Load MERT weights if they exist
        mert_path = latest_ckpt / "mert_encoder.pt"
        if mert_path.exists():
            model.load_mert_layer_weights(str(mert_path))
    else:
        logger.warning("No checkpoints found! Using random weights.")

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model on {device}")

    # Create dummy batch
    # 2 different audio samples
    # Sample 1: Silence
    audio1 = torch.zeros(1, 24000)
    # Sample 2: Noise
    audio2 = torch.randn(1, 24000)
    
    audio = torch.cat([audio1, audio2], dim=0).to(device)
    
    texts = ["The quick brown fox jumps over the lazy dog.", "Mathematical formulas are used to describe the laws of physics."]
    
    logger.info("Forward pass...")
    with torch.set_grad_enabled(True):
        outputs = model(audio, texts, return_embeddings=True)
    
    loss = outputs["loss"]
    logger.info(f"Loss: {loss.item()}")
    
    audio_pooled = outputs["audio_pooled"]
    text_pooled = outputs["text_pooled"]
    
    logger.info(f"Audio Pooled Shape: {audio_pooled.shape}")
    logger.info(f"Text Pooled Shape: {text_pooled.shape}")
    
    # Check norms
    logger.info(f"Audio Norms: {torch.norm(audio_pooled, dim=1)}")
    logger.info(f"Text Norms: {torch.norm(text_pooled, dim=1)}")

    # Check if audio embeddings are different
    diff = torch.norm(audio_pooled[0] - audio_pooled[1])
    logger.info(f"Distance between audio embeddings: {diff.item()}")
    
    if diff.item() < 1e-6:
        logger.error("Audio embeddings are identical! Mode collapse or broken input.")
    else:
        logger.info("Audio embeddings are distinct.")

    # Check if text embeddings are different
    diff_text = torch.norm(text_pooled[0] - text_pooled[1])
    logger.info(f"Distance between text embeddings: {diff_text.item()}")
    
    # Calculate cosine similarity between texts
    cos_sim = torch.nn.functional.cosine_similarity(text_pooled[0].unsqueeze(0), text_pooled[1].unsqueeze(0))
    logger.info(f"Cosine similarity between texts: {cos_sim.item()}")

    # Check gradients
    logger.info("Backward pass...")
    loss.backward()
    
    # Check gradients on QFormer
    qformer_grad_norm = 0.0
    for p in model.qformer.parameters():
        if p.grad is not None:
            qformer_grad_norm += p.grad.norm().item()
            
    logger.info(f"QFormer Grad Norm: {qformer_grad_norm}")
    
    if qformer_grad_norm == 0.0:
        logger.error("No gradients on QFormer!")

if __name__ == "__main__":
    test_model()
