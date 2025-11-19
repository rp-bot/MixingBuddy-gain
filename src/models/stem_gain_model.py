"""
Stem detection and gain regression model.

This model performs two tasks:
1. Classify which stem needs adjustment (vocals, drums, bass)
2. Predict the required gain adjustment in dB (regression)
"""

import logging
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

from src.models.encoders.encodec import EncodecEncoder
from src.models.encoders.mert import MERTEncoder
from src.models.encoders.passt import PaSSTEncoder
from src.models.projections.mlp import MLPProjection
from src.models.projections.linear import LinearProjection

logger = logging.getLogger(__name__)


class StemGainModel(nn.Module):
    """Model for stem classification and gain regression.
    
    Architecture:
    - Audio Encoder (MERT/EnCodec) -> Optional Projection -> Task Heads
    - Classification head: num_classes labels:
        0: vocals
        1: drums
        2: bass
        3: no_error (mix balanced, no adjustment)
    - Regression head: gain in dB
    """
    
    def __init__(
        self,
        encoder_config: Optional[Dict[str, Any]] = None,
        projection_config: Optional[Dict[str, Any]] = None,
        num_classes: int = 4,
        pooling_method: str = "mean",
        head_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            encoder_config: Configuration for audio encoder
            projection_config: Optional configuration for projection layer
                - If None: no projection (encoder -> heads directly)
                - If dict: create projection layer
            num_classes: Number of stem classes (default: 4 for vocals, drums, bass, no_error)
            pooling_method: How to pool temporal features ("mean", "attention", or "max")
            head_config: Optional configuration for task heads
                - If None: uses default 2-layer architecture (1024 -> 512 -> output)
                - If dict: configurable architecture with keys:
                    - hidden_dims: List of hidden dimensions (e.g., [768, 512, 256])
                    - dropout: Dropout rate (default: 0.1)
                    - activation: Activation function (default: "relu")
        """
        super().__init__()
        
        # Initialize encoder
        if encoder_config is None:
            encoder_config = {
                "model_name": "facebook/encodec_24khz",
                "freeze": True,
                "device": None,
            }
        
        encoder_model_name = encoder_config.get("model_name", "")
        if "MERT" in encoder_model_name or "m-a-p/MERT" in encoder_model_name:
            logger.info(f"Using MERT encoder: {encoder_model_name}")
            if "input_sample_rate" not in encoder_config:
                encoder_config["input_sample_rate"] = 32000
            self.audio_encoder = MERTEncoder(**encoder_config)
        elif "passt" in encoder_model_name.lower():
            logger.info(f"Using PaSST encoder: {encoder_model_name}")
            if "input_sample_rate" not in encoder_config:
                encoder_config["input_sample_rate"] = 32000
            self.audio_encoder = PaSSTEncoder(**encoder_config)
        else:
            logger.info(f"Using EnCodec encoder: {encoder_model_name}")
            self.audio_encoder = EncodecEncoder(**encoder_config)
        
        encoder_output_dim = self.audio_encoder.output_dim
        
        # Initialize optional projection
        self.use_projection = projection_config is not None
        if self.use_projection:
            projection_type = projection_config.get("type", "linear")
            if projection_type == "linear":
                self.projection = LinearProjection(
                    input_dim=encoder_output_dim,
                    output_dim=projection_config.get("output_dim", encoder_output_dim),
                    dropout=projection_config.get("dropout", 0.1),
                    activation=projection_config.get("activation", None),
                )
            elif projection_type == "mlp":
                mlp_config = dict(projection_config)
                mlp_config.pop("type", None)
                self.projection = MLPProjection(
                    input_dim=encoder_output_dim,
                    output_dim=projection_config.get("output_dim", encoder_output_dim),
                    **mlp_config
                )
            else:
                raise ValueError(f"Unsupported projection type: {projection_type}")
            feature_dim = projection_config.get("output_dim", encoder_output_dim)
        else:
            self.projection = None
            feature_dim = encoder_output_dim
        
        # Initialize pooling
        self.pooling_method = pooling_method
        if pooling_method == "attention":
            self.attention_pooling = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.Tanh(),
                nn.Linear(feature_dim, 1),
            )
        else:
            self.attention_pooling = None
        
        # Head configuration
        if head_config is None:
            head_config = {}
        
        head_hidden_dims = head_config.get("hidden_dims", [feature_dim // 2])  # Default: [512]
        head_dropout = head_config.get("dropout", 0.1)
        head_activation = head_config.get("activation", "relu")
        
        # Build activation function
        if head_activation == "relu":
            activation_fn = nn.ReLU
        elif head_activation == "gelu":
            activation_fn = nn.GELU
        elif head_activation == "tanh":
            activation_fn = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {head_activation}")
        
        # Classification head: configurable depth
        cls_layers = []
        prev_dim = feature_dim
        for hidden_dim in head_hidden_dims:
            cls_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
                nn.Dropout(head_dropout),
            ])
            prev_dim = hidden_dim
        cls_layers.append(nn.Linear(prev_dim, num_classes))
        self.classification_head = nn.Sequential(*cls_layers)
        
        # Regression head: same architecture as classification
        reg_layers = []
        prev_dim = feature_dim
        for hidden_dim in head_hidden_dims:
            reg_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
                nn.Dropout(head_dropout),
            ])
            prev_dim = hidden_dim
        reg_layers.append(nn.Linear(prev_dim, 1))
        self.regression_head = nn.Sequential(*reg_layers)
        
        # Move components to same device as encoder
        encoder_device = self.audio_encoder.device
        if self.projection is not None:
            self.projection = self.projection.to(encoder_device)
        self.classification_head = self.classification_head.to(encoder_device)
        self.regression_head = self.regression_head.to(encoder_device)
        if self.attention_pooling is not None:
            self.attention_pooling = self.attention_pooling.to(encoder_device)
        
        self.num_classes = num_classes
    
    def forward(
        self,
        audio: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        gain_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            audio: Audio tensor, shape [batch_size, num_samples] or [num_samples]
            labels: Optional classification labels, shape [batch_size]
            gain_labels: Optional regression labels, shape [batch_size]
        
        Returns:
            Dictionary with:
                - classification_logits: [batch_size, num_classes]
                - gain_prediction: [batch_size, 1]
                - loss: Combined loss (if labels provided)
        """
        # Encode audio: [batch, time_steps, feature_dim]
        audio_features = self.audio_encoder.encode(audio)
        
        # Apply projection if used
        if self.use_projection:
            audio_features = self.projection(audio_features)
        
        # Pool temporal dimension: [batch, time_steps, feature_dim] -> [batch, feature_dim]
        if self.pooling_method == "mean":
            pooled_features = audio_features.mean(dim=1)
        elif self.pooling_method == "max":
            pooled_features = audio_features.max(dim=1)[0]
        elif self.pooling_method == "attention":
            # Attention weights: [batch, time_steps, 1]
            attention_weights = self.attention_pooling(audio_features)
            attention_weights = torch.softmax(attention_weights, dim=1)
            # Weighted sum: [batch, feature_dim]
            pooled_features = (audio_features * attention_weights).sum(dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")
        
        # Classification head
        classification_logits = self.classification_head(pooled_features)
        
        # Regression head
        gain_prediction = self.regression_head(pooled_features)
        
        output = {
            "classification_logits": classification_logits,
            "gain_prediction": gain_prediction,
        }
        
        # Compute loss if labels provided
        if labels is not None or gain_labels is not None:
            loss = torch.tensor(0.0, device=audio_features.device, dtype=audio_features.dtype)
            
            if labels is not None:
                cls_loss = nn.functional.cross_entropy(classification_logits, labels)
                loss = loss + cls_loss
                output["classification_loss"] = cls_loss
            
            if gain_labels is not None:
                reg_loss = nn.functional.mse_loss(gain_prediction.squeeze(-1), gain_labels)
                loss = loss + reg_loss
                output["regression_loss"] = reg_loss
            
            output["loss"] = loss
        
        return output
    
    def print_trainable_parameters(self):
        """Print number of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

