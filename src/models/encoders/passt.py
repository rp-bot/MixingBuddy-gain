"""
PaSST (Patchout Audio Spectrogram Transformer) encoder.

PaSST is a pre-trained audio transformer model that extracts powerful
audio representations from raw waveforms.
"""

import torch
import torch.nn as nn
from typing import Optional, Union
import torchaudio.transforms as T
import logging
import warnings
import sys
import os

logger = logging.getLogger(__name__)

# Suppress debug prints from hear21passt library
# Redirect stdout temporarily when loading the model to suppress prints
class SuppressOutput:
    """Context manager to suppress stdout/stderr."""
    def __init__(self, suppress_stdout=True, suppress_stderr=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._original_stdout = None
        self._original_stderr = None

    def __enter__(self):
        if self.suppress_stdout:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        if self.suppress_stderr:
            self._original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress_stdout:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        if self.suppress_stderr:
            sys.stderr.close()
            sys.stderr = self._original_stderr

try:
    from hear21passt.base import get_basic_model
    HEAR21PASST_AVAILABLE = True
except ImportError:
    HEAR21PASST_AVAILABLE = False
    logger.warning("hear21passt not available. PaSST encoder cannot be used.")


class PaSSTEncoder(nn.Module):
    """PaSST audio encoder that extracts features from pre-trained transformer.
    
    PaSST expects 32kHz audio input and outputs sequence embeddings.
    This encoder:
    1. Resamples input audio to 32kHz (PaSST's expected sample rate)
    2. Extracts timestamp embeddings using get_timestamp_embeddings
    3. Returns features of shape [batch, time_steps, 768]
    """

    def __init__(
        self,
        model_name: str = "hear21passt",
        freeze: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        input_sample_rate: int = 32000,
    ):
        super().__init__()
        
        if not HEAR21PASST_AVAILABLE:
            raise ImportError(
                "hear21passt is not installed. Please install it with: pip install hear21passt"
            )
        
        # Load PaSST model
        # mode="logits" gives us the full model with classification head
        # We'll use get_timestamp_embeddings to get sequence features
        # Suppress verbose output during model loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            with SuppressOutput(suppress_stdout=True, suppress_stderr=False):
                self.passt_model = get_basic_model(mode="logits")
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.passt_model = self.passt_model.to(self.device)
        
        if freeze:
            self.passt_model.eval()
            for param in self.passt_model.parameters():
                param.requires_grad = False
        else:
            # Enable gradient checkpointing if trainable
            if hasattr(self.passt_model, "gradient_checkpointing_enable"):
                self.passt_model.gradient_checkpointing_enable()
        
        self.frozen = freeze
        self.model_name = model_name
        self.input_sample_rate = input_sample_rate
        
        # PaSST expects 32kHz audio
        passt_sample_rate = 32000
        if input_sample_rate != passt_sample_rate:
            self.resampler = T.Resample(
                orig_freq=input_sample_rate, new_freq=passt_sample_rate
            )
            self.resampler = self.resampler.to(self.device)
            self.needs_resampling = True
        else:
            self.resampler = None
            self.needs_resampling = False

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio and return timestamp embeddings.
        
        Args:
            audio: Input audio tensor, shape [batch_size, num_samples] or [num_samples]
                   Expected to be at self.input_sample_rate
        
        Returns:
            features: Timestamp embeddings, shape [batch, time_steps, 768]
        """
        # Handle single sample case
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Ensure audio is on the correct device
        audio = audio.to(self.device)
        
        # Resample if needed
        if self.needs_resampling:
            audio = self.resampler(audio)
        
        # PaSST expects audio at 32kHz
        # Extract timestamp embeddings (sequence of features)
        # The wrapper has model.mel (mel spectrogram) and model.net (transformer)
        # We need to get sequence features from the transformer before the classification head
        
        # Suppress debug prints during inference
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            with SuppressOutput(suppress_stdout=True, suppress_stderr=False):
                # Extract mel spectrogram
                mel = self.passt_model.mel(audio)
                
                # Ensure mel has 4 dimensions [batch, channels, freq, time]
                # mel might be 3D [batch, freq, time], so add channel dimension
                if mel.dim() == 3:
                    mel = mel.unsqueeze(1)  # Add channel dimension: [batch, 1, freq, time]
                
                # Get features from the transformer network
                # model.net is the actual PaSST transformer
                # We'll manually forward through to get sequence embeddings
                if self.frozen:
                    with torch.no_grad():
                        # Patch embedding
                        x = self.passt_model.net.patch_embed(mel)
                        B, C, H, W = x.shape
                        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
                        
                        # Add position embeddings (if they exist)
                        if hasattr(self.passt_model.net, 'pos_embed') and self.passt_model.net.pos_embed is not None:
                            # Interpolate position embeddings if needed
                            pos_embed = self.passt_model.net.pos_embed
                            if pos_embed.shape[1] != H * W:
                                # Need to interpolate - this is complex, try without for now
                                pass
                            else:
                                x = x + pos_embed
                        elif hasattr(self.passt_model.net, 'time_new_pos_embed') and hasattr(self.passt_model.net, 'freq_new_pos_embed'):
                            # PaSST uses separate time and frequency position embeddings
                            time_pos = self.passt_model.net.time_new_pos_embed[:, :, :, :W]
                            freq_pos = self.passt_model.net.freq_new_pos_embed[:, :, :H, :]
                            pos_embed = time_pos + freq_pos
                            pos_embed = pos_embed.flatten(2).transpose(1, 2)
                            x = x + pos_embed
                        
                        x = self.passt_model.net.pos_drop(x)
                        
                        # Add cls and dist tokens
                        if hasattr(self.passt_model.net, 'cls_token') and self.passt_model.net.cls_token is not None:
                            cls_token = self.passt_model.net.cls_token.expand(B, -1, -1)
                            if hasattr(self.passt_model.net, 'dist_token') and self.passt_model.net.dist_token is not None:
                                dist_token = self.passt_model.net.dist_token.expand(B, -1, -1)
                                x = torch.cat((cls_token, dist_token, x), dim=1)
                            else:
                                x = torch.cat((cls_token, x), dim=1)
                        
                        # Forward through transformer blocks
                        for blk in self.passt_model.net.blocks:
                            x = blk(x)
                        
                        # Apply norm
                        x = self.passt_model.net.norm(x)
                        
                        # Remove cls and dist tokens if they were added
                        if hasattr(self.passt_model.net, 'cls_token') and self.passt_model.net.cls_token is not None:
                            if hasattr(self.passt_model.net, 'dist_token') and self.passt_model.net.dist_token is not None:
                                features = x[:, 2:, :]  # Skip cls and dist tokens
                            else:
                                features = x[:, 1:, :]  # Skip cls token
                        else:
                            features = x
                else:
                    # Same but without no_grad
                    # Ensure mel has 4 dimensions
                    if mel.dim() == 3:
                        mel = mel.unsqueeze(1)
                    x = self.passt_model.net.patch_embed(mel)
                    B, C, H, W = x.shape
                    x = x.flatten(2).transpose(1, 2)
                    
                    if hasattr(self.passt_model.net, 'time_new_pos_embed') and hasattr(self.passt_model.net, 'freq_new_pos_embed'):
                        time_pos = self.passt_model.net.time_new_pos_embed[:, :, :, :W]
                        freq_pos = self.passt_model.net.freq_new_pos_embed[:, :, :H, :]
                        pos_embed = time_pos + freq_pos
                        pos_embed = pos_embed.flatten(2).transpose(1, 2)
                        x = x + pos_embed
                    
                    x = self.passt_model.net.pos_drop(x)
                    
                    if hasattr(self.passt_model.net, 'cls_token') and self.passt_model.net.cls_token is not None:
                        cls_token = self.passt_model.net.cls_token.expand(B, -1, -1)
                        if hasattr(self.passt_model.net, 'dist_token') and self.passt_model.net.dist_token is not None:
                            dist_token = self.passt_model.net.dist_token.expand(B, -1, -1)
                            x = torch.cat((cls_token, dist_token, x), dim=1)
                        else:
                            x = torch.cat((cls_token, x), dim=1)
                    
                    for blk in self.passt_model.net.blocks:
                        x = blk(x)
                    
                    x = self.passt_model.net.norm(x)
                    
                    if hasattr(self.passt_model.net, 'cls_token') and self.passt_model.net.cls_token is not None:
                        if hasattr(self.passt_model.net, 'dist_token') and self.passt_model.net.dist_token is not None:
                            features = x[:, 2:, :]
                        else:
                            features = x[:, 1:, :]
                    else:
                        features = x
                
                # Ensure features are on the correct device
                features = features.to(self.device)
        
        return features

    @property
    def output_dim(self) -> int:
        """Output feature dimension: 768 (PaSST embedding dimension)"""
        return 768

    @property
    def sample_rate(self) -> int:
        """Sample rate expected by the PaSST model (32kHz)"""
        return 32000

    @property
    def hop_length(self) -> int:
        """Effective hop length (stride) at the input sample rate.
        
        PaSST produces features at approximately 10 Hz (10 frames per second) at 32kHz.
        This property returns the hop length in terms of the input sample rate.
        
        Returns:
            int: Number of input samples per output feature frame
        """
        # PaSST produces approximately 10 Hz features at 32kHz input
        # Feature rate is determined by the patch embedding stride
        # At 32kHz: hop_length = 32000 / 10 = 3200
        # At input_sample_rate: hop_length = input_sample_rate / 10
        passt_feature_rate_hz = 10  # Approximate feature rate of PaSST
        return int(self.input_sample_rate / passt_feature_rate_hz)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encode(audio)

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "frozen": self.frozen,
            "output_dim": self.output_dim,
            "sample_rate": self.sample_rate,
            "input_sample_rate": self.input_sample_rate,
            "needs_resampling": self.needs_resampling,
            "device": str(self.device),
        }


def create_passt_encoder(
    model_name: str = "hear21passt",
    freeze: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    input_sample_rate: int = 32000,
) -> PaSSTEncoder:
    return PaSSTEncoder(
        model_name=model_name,
        freeze=freeze,
        device=device,
        input_sample_rate=input_sample_rate,
    )

