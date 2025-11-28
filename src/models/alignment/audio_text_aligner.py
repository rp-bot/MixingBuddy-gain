"""
Audio-Text Alignment Model for pre-training projection layers.

This module provides a model that aligns audio embeddings from MERT 
to text embeddings from an LLM's embedding layer using a Q-Former architecture.
The pre-trained Q-Former can later be used as a projection layer in the full
multimodal LLM pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import logging

from transformers import AutoTokenizer, AutoModel

from ..encoders.mert import MERTEncoder
from ..projections.qformer import QFormerProjection

logger = logging.getLogger(__name__)


class AttentionPooling(nn.Module):
    """Learnable attention pooling over sequence."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1, bias=False),
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden]
            mask: [batch, seq_len] - True for positions to MASK OUT (padding)
        """
        weights = self.attention(x)  # [batch, seq_len, 1]
        if mask is not None:
            # mask is True for padding positions - set those to -inf
            weights = weights.masked_fill(mask.unsqueeze(-1), float('-inf'))
        weights = torch.softmax(weights, dim=1)
        # Handle case where all positions are masked (results in NaN from softmax)
        weights = torch.nan_to_num(weights, nan=0.0)
        return (x * weights).sum(dim=1)


class AudioTextAligner(nn.Module):
    """Audio-Text alignment model for pre-training Q-Former projection.
    
    Architecture:
        audio -> MERT (frozen) -> Q-Former (trainable) -> audio_embeds [B, N, hidden]
                                                                |
                                                          Alignment Loss
                                                                |
        text -> Tokenizer -> LLM Embed (frozen) -> text_embeds [B, T, hidden] -> Pool
    
    The model learns to align audio representations with text representations
    in the LLM's embedding space.
    """
    
    def __init__(
        self,
        encoder_config: Dict[str, Any],
        qformer_config: Dict[str, Any],
        llm_model_name: str = "Qwen/Qwen2-7B-Instruct",
        loss_config: Optional[Dict[str, Any]] = None,
        pooling_type: str = "attention",  # NEW parameter
    ):
        """Initialize the Audio-Text Aligner.
        
        Args:
            encoder_config: Configuration for MERT encoder
            qformer_config: Configuration for Q-Former projection
            llm_model_name: Name of the LLM to get embeddings from
            loss_config: Configuration for loss computation
                - mse_weight: Weight for MSE loss (default: 1.0)
                - contrastive_weight: Weight for contrastive loss (default: 0.1)
                - temperature: Temperature for contrastive loss (default: 0.07)
        """
        super().__init__()
        
        self.llm_model_name = llm_model_name
        
        # Default loss config
        if loss_config is None:
            loss_config = {}
        self.mse_weight = loss_config.get("mse_weight", 1.0)
        self.cosine_weight = loss_config.get("cosine_weight", 0.0)
        self.contrastive_weight = loss_config.get("contrastive_weight", 0.1)
        self.temperature = loss_config.get("temperature", 0.07)
        
        # Initialize MERT encoder (frozen)
        logger.info("Initializing MERT encoder...")
        encoder_config = dict(encoder_config)
        encoder_config["freeze"] = True  # Always freeze for alignment
        self.audio_encoder = MERTEncoder(**encoder_config)
        
        # Get LLM hidden size for Q-Former output
        logger.info(f"Loading LLM embedding layer from {llm_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Only load the embedding layer, not the full LLM
        llm = AutoModel.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16,
        )
        self.llm_embed = llm.get_input_embeddings()
        self.llm_hidden_size = self.llm_embed.weight.shape[1]
        
        # Freeze LLM embeddings
        for param in self.llm_embed.parameters():
            param.requires_grad = False
        
        # Delete the rest of the LLM to save memory
        del llm
        torch.cuda.empty_cache()
        
        # Initialize Q-Former projection (trainable)
        logger.info("Initializing Q-Former projection...")
        qformer_config = dict(qformer_config)
        qformer_config["audio_dim"] = self.audio_encoder.output_dim
        qformer_config["output_dim"] = self.llm_hidden_size
        self.qformer = QFormerProjection(**qformer_config)
        
        self.pooling_type = pooling_type
        if pooling_type == "attention":
            self.audio_pooler = AttentionPooling(self.llm_hidden_size)
            self.text_pooler = AttentionPooling(self.llm_hidden_size)
        elif pooling_type == "gem":
            raise NotImplementedError("GeM pooling not yet implemented. Use 'attention', 'mean', 'max', or 'cls'.")
        # mean/max/cls use the existing pool_embeddings method
        
        # Move poolers to the same device as the model
        if hasattr(self, 'audio_pooler'):
            device = next(self.audio_encoder.parameters()).device
            self.audio_pooler = self.audio_pooler.to(device)
            self.text_pooler = self.text_pooler.to(device)
        
        logger.info(f"AudioTextAligner initialized:")
        logger.info(f"  Audio encoder output dim: {self.audio_encoder.output_dim}")
        logger.info(f"  LLM hidden size: {self.llm_hidden_size}")
        logger.info(f"  Q-Former num queries: {self.qformer.num_queries}")
        logger.info(f"  Q-Former parameters: {sum(p.numel() for p in self.qformer.parameters()):,}")
    
    def get_text_embeddings(
        self,
        texts: list[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get text embeddings from LLM embedding layer.
        
        Args:
            texts: List of text strings
            device: Device to put tensors on
            
        Returns:
            Tuple of:
                - text_embeds: [batch, max_len, hidden_size]
                - attention_mask: [batch, max_len]
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Get embeddings (frozen)
        with torch.no_grad():
            text_embeds = self.llm_embed(input_ids)
        
        return text_embeds, attention_mask
    
    def pool_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Pool sequence embeddings to a single vector.
        
        Args:
            embeddings: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] - 1 for real tokens, 0 for padding
            pooling: Pooling method ('mean', 'max', 'cls')
            
        Returns:
            Pooled embeddings: [batch, hidden_size]
        """
        if pooling == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask = attention_mask.unsqueeze(-1).float()
                masked_embeds = embeddings * mask
                pooled = masked_embeds.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = embeddings.mean(dim=1)
        elif pooling == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                masked_embeds = embeddings * mask + (1 - mask) * (-1e9)
                pooled = masked_embeds.max(dim=1)[0]
            else:
                pooled = embeddings.max(dim=1)[0]
        elif pooling == "cls":
            pooled = embeddings[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        return pooled
    
    def compute_contrastive_loss(
        self,
        audio_pooled: torch.Tensor,
        text_pooled: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss.
        
        Args:
            audio_pooled: [batch, hidden_size]
            text_pooled: [batch, hidden_size]
            
        Returns:
            Contrastive loss scalar
        """
        # Normalize embeddings with eps for numerical stability
        audio_norm = F.normalize(audio_pooled, p=2, dim=-1, eps=1e-8)
        text_norm = F.normalize(text_pooled, p=2, dim=-1, eps=1e-8)
        
        # Compute similarity matrix with temperature clamping for stability
        temperature = max(self.temperature, 1e-8)
        similarity = torch.matmul(audio_norm, text_norm.T) / temperature
        
        # Clamp to prevent overflow in cross_entropy softmax
        similarity = torch.clamp(similarity, min=-100.0, max=100.0)
        
        # Labels: diagonal elements are positive pairs
        batch_size = audio_pooled.shape[0]
        labels = torch.arange(batch_size, device=audio_pooled.device)
        
        # Cross-entropy loss in both directions
        loss_a2t = F.cross_entropy(similarity, labels)
        loss_t2a = F.cross_entropy(similarity.T, labels)
        
        return (loss_a2t + loss_t2a) / 2
    
    def forward(
        self,
        audio: torch.Tensor,
        texts: list[str],
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing alignment loss.
        
        Args:
            audio: Audio waveforms [batch, samples]
            texts: List of target text strings
            return_embeddings: If True, also return the embeddings
            
        Returns:
            Dictionary with:
                - loss: Total alignment loss
                - mse_loss: MSE component
                - contrastive_loss: Contrastive component (if weight > 0)
                - audio_embeds: (optional) Q-Former outputs
                - text_embeds: (optional) Text embeddings
        """
        device = audio.device
        
        # Get audio embeddings through MERT + Q-Former
        audio_features = self.audio_encoder(audio)
        audio_embeds = self.qformer(audio_features)  # [batch, num_queries, hidden]
        
        # Get text embeddings from LLM
        text_embeds, text_mask = self.get_text_embeddings(texts, device)
        
        # Ensure matching dtype - align text (frozen) to audio (trainable)
        # This prevents issues where audio_embeds becomes fp16 while pooler is fp32
        if text_embeds.dtype != audio_embeds.dtype:
            text_embeds = text_embeds.to(audio_embeds.dtype)
        
        # Pool to single vectors
        if self.pooling_type in ["attention", "gem"]:
            audio_pooled = self.audio_pooler(audio_embeds)
            text_pooled = self.text_pooler(text_embeds, (text_mask == 0))  # Invert mask
        else:
            audio_pooled = self.pool_embeddings(audio_embeds, pooling=self.pooling_type)
            text_pooled = self.pool_embeddings(text_embeds, text_mask, pooling=self.pooling_type)
        
        # Compute MSE loss
        mse_loss = F.mse_loss(audio_pooled, text_pooled) if self.mse_weight > 0 else torch.tensor(0.0, device=device)
        
        # Compute cosine similarity loss
        if self.cosine_weight > 0:
            cosine_sim = F.cosine_similarity(audio_pooled, text_pooled, dim=-1, eps=1e-8)
            cosine_loss = (1 - cosine_sim).mean()
        else:
            cosine_loss = torch.tensor(0.0, device=device)
        
        # Compute contrastive loss
        if self.contrastive_weight > 0:
            contrastive_loss = self.compute_contrastive_loss(audio_pooled, text_pooled)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = (
            self.mse_weight * mse_loss + 
            self.cosine_weight * cosine_loss + 
            self.contrastive_weight * contrastive_loss
        )
        
        result = {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "cosine_loss": cosine_loss,
            "contrastive_loss": contrastive_loss,
        }
        
        if return_embeddings:
            result["audio_embeds"] = audio_embeds
            result["text_embeds"] = text_embeds
            result["audio_pooled"] = audio_pooled
            result["text_pooled"] = text_pooled
        
        return result
    
    def save_trainable_weights(self, path: str):
        """Save all trainable weights (Q-Former, Poolers, MERT layer weights).
        
        Args:
            path: Path to save the state dict
        """
        state_dict = {
            "qformer": self.qformer.state_dict(),
        }
        
        # Save poolers if they exist
        if hasattr(self, "audio_pooler"):
            state_dict["audio_pooler"] = self.audio_pooler.state_dict()
        if hasattr(self, "text_pooler"):
            state_dict["text_pooler"] = self.text_pooler.state_dict()
            
        # Save MERT layer weights if they exist
        if hasattr(self.audio_encoder, "layer_weights"):
            state_dict["mert_layer_weights"] = self.audio_encoder.layer_weights
            
        torch.save(state_dict, path)
        logger.info(f"Saved trainable weights to {path}")

    def load_trainable_weights(self, path: str):
        """Load all trainable weights.
        
        Args:
            path: Path to the state dict file
        """
        state_dict = torch.load(path, map_location="cpu")
        
        # Load Q-Former
        if "qformer" in state_dict:
            self.qformer.load_state_dict(state_dict["qformer"])
        else:
            # Fallback for old format where the file was just the qformer dict
            try:
                self.qformer.load_state_dict(state_dict)
                logger.info("Loaded legacy Q-Former weights")
                return
            except:
                logger.warning("Could not load Q-Former weights")
        
        # Load poolers
        if "audio_pooler" in state_dict and hasattr(self, "audio_pooler"):
            self.audio_pooler.load_state_dict(state_dict["audio_pooler"])
        if "text_pooler" in state_dict and hasattr(self, "text_pooler"):
            self.text_pooler.load_state_dict(state_dict["text_pooler"])
            
        # Load MERT layer weights
        if "mert_layer_weights" in state_dict and hasattr(self.audio_encoder, "layer_weights"):
            self.audio_encoder.layer_weights.data = state_dict["mert_layer_weights"].to(
                self.audio_encoder.layer_weights.device
            )
            
        logger.info(f"Loaded trainable weights from {path}")

    def save_qformer(self, path: str):
        """Save only the Q-Former weights for later use.
        
        Args:
            path: Path to save the Q-Former state dict
        """
        torch.save(self.qformer.state_dict(), path)
        logger.info(f"Saved Q-Former weights to {path}")
    
    def save_mert_layer_weights(self, path: str):
        """Save MERT layer weights for later use.
        
        Saves in the same format as 06_train_model.py (full encoder state dict)
        for compatibility.
        
        Args:
            path: Path to save the layer weights
        """
        if hasattr(self.audio_encoder, "layer_weights"):
            # Save full encoder state dict for compatibility with 06_train_model.py
            torch.save(self.audio_encoder.state_dict(), path)
            logger.info(f"Saved MERT encoder state (including layer weights) to {path}")
        else:
            logger.warning("MERT encoder does not have learnable layer weights")
    
    def load_qformer(self, path: str):
        """Load Q-Former weights from a checkpoint.
        
        Args:
            path: Path to the Q-Former state dict
        """
        self.qformer.load_state_dict(torch.load(path))
        logger.info(f"Loaded Q-Former weights from {path}")
    
    def load_mert_layer_weights(self, path: str):
        """Load MERT layer weights from a checkpoint.
        
        Supports both formats:
        1. Full encoder state dict (from 06_train_model.py)
        2. Simple {"layer_weights": tensor} format
        
        Args:
            path: Path to the layer weights file
        """
        if hasattr(self.audio_encoder, "layer_weights"):
            state = torch.load(path, map_location="cpu")
            if "layer_weights" in state:
                # Simple format
                self.audio_encoder.layer_weights.data = state["layer_weights"].to(
                    self.audio_encoder.layer_weights.device
                )
            else:
                # Full encoder state dict format
                self.audio_encoder.load_state_dict(state, strict=False)
            logger.info(f"Loaded MERT layer weights from {path}")
        else:
            logger.warning("MERT encoder does not have learnable layer weights")
    
    def print_trainable_parameters(self):
        """Print parameter counts by component."""
        # MERT encoder
        mert_params = sum(p.numel() for p in self.audio_encoder.parameters())
        mert_trainable = sum(
            p.numel() for p in self.audio_encoder.parameters() if p.requires_grad
        )
        
        # Q-Former
        qformer_params = sum(p.numel() for p in self.qformer.parameters())
        qformer_trainable = sum(
            p.numel() for p in self.qformer.parameters() if p.requires_grad
        )
        
        # LLM embed
        embed_params = sum(p.numel() for p in self.llm_embed.parameters())
        embed_trainable = sum(
            p.numel() for p in self.llm_embed.parameters() if p.requires_grad
        )
        
        total_params = mert_params + qformer_params + embed_params
        total_trainable = mert_trainable + qformer_trainable + embed_trainable
        
        logger.info("=" * 60)
        logger.info("Parameter Summary:")
        logger.info(f"  MERT Encoder:    {mert_params:>12,} ({mert_trainable:,} trainable)")
        logger.info(f"  Q-Former:        {qformer_params:>12,} ({qformer_trainable:,} trainable)")
        logger.info(f"  LLM Embeddings:  {embed_params:>12,} ({embed_trainable:,} trainable)")
        logger.info("-" * 60)
        logger.info(f"  Total:           {total_params:>12,} ({total_trainable:,} trainable)")
        logger.info(f"  Trainable:       {100 * total_trainable / total_params:.2f}%")
        logger.info("=" * 60)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "encoder": self.audio_encoder.get_model_info(),
            "qformer": self.qformer.get_model_info(),
            "llm_model_name": self.llm_model_name,
            "llm_hidden_size": self.llm_hidden_size,
            "loss_config": {
                "mse_weight": self.mse_weight,
                "contrastive_weight": self.contrastive_weight,
                "temperature": self.temperature,
            },
        }

