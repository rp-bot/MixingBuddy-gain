"""
Encodec encoder wrapper for converting audio to features.
"""

import warnings
import torch
import torch.nn as nn
from typing import Optional, Union

# Suppress deprecation warnings from Encodec's use of old PyTorch APIs
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.nn.utils.weight_norm"
)

try:
    from encodec import EncodecModel

    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False
    EncodecModel = None


class EncodecEncoder(nn.Module):
    """Encodec encoder wrapper for audio feature extraction.

    This class wraps the Facebook Encodec model to provide a simple interface
    for encoding audio to continuous features suitable for multimodal models.
    """

    def __init__(
        self,
        model_name: str = "facebook/encodec_24khz",
        target_bandwidth: float = 6.0,
        freeze: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize the Encodec encoder.

        Args:
            model_name: Name of the Encodec model to use
            target_bandwidth: Target bandwidth for encoding (higher = better quality)
            freeze: Whether to freeze the model parameters
            device: Device to run the model on
        """
        super().__init__()

        if not ENCODEC_AVAILABLE:
            raise ImportError(
                "Encodec is not available. Please install it with: pip install encodec"
            )

        # Load the model
        if model_name == "facebook/encodec_24khz":
            self.model = EncodecModel.encodec_model_24khz()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Set target bandwidth
        self.model.set_target_bandwidth(target_bandwidth)

        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # Freeze parameters if requested
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.frozen = freeze
        self.target_bandwidth = target_bandwidth

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to continuous features.

        Args:
            audio: Input audio tensor of shape (batch_size, samples) or (samples,)
                  Expected sample rate: 24kHz

        Returns:
            Continuous features of shape (batch_size, time_steps, hidden_dim)
            where hidden_dim is typically 128 for Encodec
        """
        # Ensure audio is on the correct device
        audio = audio.to(self.device)

        # Handle single sample case
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Add channel dimension if needed (Encodec expects mono audio)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # (batch, 1, samples)

        # Encode with no gradients if frozen
        if self.frozen:
            with torch.no_grad():
                encoded = self.model.encode(audio)
        else:
            encoded = self.model.encode(audio)

        # Extract continuous embeddings (not discrete codes)
        # Encodec returns a list containing a tuple: [(continuous_features, codes, bandwidth)]
        if isinstance(encoded, list) and len(encoded) > 0:
            # Get the first (and usually only) item from the list
            encoded_tuple = encoded[0]
            if isinstance(encoded_tuple, tuple) and len(encoded_tuple) > 0:
                continuous_features = encoded_tuple[
                    0
                ]  # (batch, channels, time, features)
            else:
                continuous_features = encoded_tuple
        elif isinstance(encoded, tuple):
            continuous_features = encoded[0]  # (batch, channels, time, features)
        else:
            continuous_features = encoded

        # Handle different possible shapes
        if hasattr(continuous_features, "dim") and continuous_features.dim() == 3:
            # Encodec returns (batch, channels, time_features)
            # We want (batch, time, channels, features_per_channel)
            batch_size, channels, time_features = continuous_features.shape

            # Group features per channel (Encodec standard: 75 features per channel)
            features_per_channel = 75
            remainder = time_features % features_per_channel
            if remainder != 0:
                # Trim tail so time_features is divisible by features_per_channel
                trimmed = time_features - remainder
                if trimmed <= 0:
                    raise ValueError(
                        f"Encodec features too short to reshape: time_features={time_features}, remainder={remainder}"
                    )
                continuous_features = continuous_features[:, :, :trimmed]
                time_features = trimmed

            time_steps = time_features // features_per_channel
            # Reshape to (batch, time, channels, features_per_channel)
            continuous_features = continuous_features.permute(
                0, 2, 1
            )  # (batch, time_features, channels)
            continuous_features = continuous_features.reshape(
                batch_size, time_steps, channels, features_per_channel
            )
        elif hasattr(continuous_features, "dim") and continuous_features.dim() == 4:
            # Already in correct shape: (batch, time, channels, features)
            pass
        else:
            raise ValueError(
                f"Unexpected continuous_features type/shape: {type(continuous_features)}, {getattr(continuous_features, 'shape', 'no shape')}"
            )

        return continuous_features

    @property
    def output_dim(self) -> int:
        """Get the output feature dimension."""
        # For Encodec 24kHz with 6.0 bandwidth, this is typically 8 channels * 75 features = 600
        # But we keep them separate, so this returns the feature dimension per channel
        return 75

    @property
    def output_channels(self) -> int:
        """Get the number of output channels."""
        return 8

    @property
    def sample_rate(self) -> int:
        """Get the expected input sample rate."""
        return 24000  # Encodec 24kHz model

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass (alias for encode)."""
        return self.encode(audio)

    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "model_name": "facebook/encodec_24khz",
            "target_bandwidth": self.target_bandwidth,
            "frozen": self.frozen,
            "output_dim": self.output_dim,
            "output_channels": self.output_channels,
            "sample_rate": self.sample_rate,
            "device": str(self.device),
        }


def create_encodec_encoder(
    model_name: str = "facebook/encodec_24khz",
    target_bandwidth: float = 6.0,
    freeze: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> EncodecEncoder:
    """Factory function to create an Encodec encoder.

    Args:
        model_name: Name of the Encodec model to use
        target_bandwidth: Target bandwidth for encoding
        freeze: Whether to freeze the model parameters
        device: Device to run the model on

    Returns:
        Initialized EncodecEncoder instance
    """
    return EncodecEncoder(
        model_name=model_name,
        target_bandwidth=target_bandwidth,
        freeze=freeze,
        device=device,
    )
