import torch
import torch.nn as nn
from typing import Optional, Union

from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torchaudio.transforms as T


class MERTEncoder(nn.Module):
    """MERT audio encoder that extracts features from all transformer layers.

    MERT-v1-330M outputs 25 layers of representations. This encoder:
    1. Resamples input audio from 32kHz to 24kHz (MERT's expected sample rate)
    2. Extracts all 25 hidden states
    3. Concatenates them along the feature dimension: [batch, time_steps, 25*1024]
    """

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-330M",
        freeze: bool = True,
        freeze_layer_weights: bool = False,
        unfreeze_top_n_layers: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        input_sample_rate: int = 32000,
    ):
        super().__init__()
        # Load MERT model and processor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Selectively unfreeze top N layers if requested
            if unfreeze_top_n_layers > 0:
                # MERT is based on Wav2Vec2 architecture
                # The encoder layers are in model.encoder.layers
                if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
                    total_layers = len(self.model.encoder.layers)
                    layers_to_unfreeze = min(unfreeze_top_n_layers, total_layers)
                    
                    # Unfreeze the top N layers (higher indices = later/top layers)
                    for i in range(total_layers - layers_to_unfreeze, total_layers):
                        for param in self.model.encoder.layers[i].parameters():
                            param.requires_grad = True
                    
                    # Set model to train mode for the unfrozen layers
                    self.model.train()
                    
                    print(f"Unfroze top {layers_to_unfreeze} layers (layers {total_layers - layers_to_unfreeze} to {total_layers - 1}) of MERT encoder")
                else:
                    print(f"Warning: Could not find encoder layers to unfreeze. Model structure: {type(self.model)}")
        else:
            # Enable gradient checkpointing only if model is trainable
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()

        self.frozen = freeze
        self.unfreeze_top_n_layers = unfreeze_top_n_layers
        self.model_name = model_name
        self.input_sample_rate = input_sample_rate

        # Create resampler if needed
        mert_sample_rate = self.processor.sampling_rate
        if input_sample_rate != mert_sample_rate:
            self.resampler = T.Resample(
                orig_freq=input_sample_rate, new_freq=mert_sample_rate
            )
            self.needs_resampling = True
        else:
            self.resampler = None
            self.needs_resampling = False

        # Learnable layer weights for weighted average (25 layers + 1 embedding layer)
        # Initialize with uniform weights that sum to 1
        num_layers = 25  # MERT-v1-330M has 25 layers
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Optionally freeze layer weights
        self.freeze_layer_weights = freeze_layer_weights
        if freeze_layer_weights:
            self.layer_weights.requires_grad = False

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio and return weighted average of all layer features.

        Args:
            audio: Input audio tensor, shape [batch_size, num_samples] or [num_samples]
                   Expected to be at self.input_sample_rate

        Returns:
            features: Weighted average of layer features, shape [batch, time_steps, 1024]
        """
        # Handle single sample case
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Resample if needed
        if self.needs_resampling:
            # Resample on the same device as input audio
            audio = self.resampler(audio)

        # Prepare inputs using processor
        # Processor expects audio on CPU
        inputs = self.processor(
            raw_speech=audio.cpu(),
            sampling_rate=self.processor.sampling_rate,
            return_tensors="pt",
        )

        # Move inputs to model device and ensure correct shape
        input_values = inputs.input_values.to(self.device)

        # Ensure input_values is 2D: [batch_size, sequence_length]
        if input_values.dim() == 3:
            input_values = input_values.squeeze(1)  # Remove extra dimension if present

        # Extract features with all hidden states
        # Don't use no_grad if we have unfrozen layers
        if self.frozen and self.unfreeze_top_n_layers == 0:
            with torch.no_grad():
                outputs = self.model(
                    input_values,
                    output_hidden_states=True,
                )
        else:
            outputs = self.model(
                input_values,
                output_hidden_states=True,
            )

        # outputs.hidden_states is a tuple of (num_layers) tensors
        # Each tensor has shape [batch, time_steps, 1024]
        # Stack all layers: [num_layers, batch, time_steps, 1024]
        all_layer_hidden_states = torch.stack(outputs.hidden_states)
        # Shape: [25, batch, time_steps, 1024]

        # Normalize layer weights using softmax to ensure they sum to 1
        normalized_weights = torch.softmax(self.layer_weights, dim=0)

        # Apply weighted average across layers
        # Reshape weights for broadcasting: [25, 1, 1, 1]
        weights_expanded = normalized_weights.view(-1, 1, 1, 1)

        # Weighted sum: [25, batch, time_steps, 1024] * [25, 1, 1, 1] -> [25, batch, time_steps, 1024]
        weighted_layers = all_layer_hidden_states * weights_expanded

        # Sum across layer dimension: [batch, time_steps, 1024]
        features = weighted_layers.sum(dim=0)

        return features

    @property
    def output_dim(self) -> int:
        """Output feature dimension: 1024 (weighted average of 25 layers)"""
        # MERT-v1-330M outputs 1024-dimensional features per layer
        # We compute a learnable weighted average across all 25 layers
        return 1024

    @property
    def sample_rate(self) -> int:
        """Sample rate expected by the MERT model (24kHz)"""
        return self.processor.sampling_rate

    @property
    def hop_length(self) -> int:
        """Effective hop length (stride) at the input sample rate.

        MERT produces ~75 Hz features (75 frames per second) at 24kHz.
        This property returns the hop length in terms of the input sample rate.

        Returns:
            int: Number of input samples per output feature frame
        """
        # MERT produces approximately 75 Hz features at 24kHz input
        # Feature rate is determined by the model architecture
        # At 24kHz: hop_length = 24000 / 75 = 320
        # At input_sample_rate: hop_length = input_sample_rate / 75
        mert_feature_rate_hz = 75  # Approximate feature rate of MERT
        return int(self.input_sample_rate / mert_feature_rate_hz)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encode(audio)

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "frozen": self.frozen,
            "unfreeze_top_n_layers": self.unfreeze_top_n_layers,
            "output_dim": self.output_dim,
            "sample_rate": self.sample_rate,
            "input_sample_rate": self.input_sample_rate,
            "needs_resampling": self.needs_resampling,
            "device": str(self.device),
            "layer_aggregation": "learnable_weighted_average",
            "num_layer_weights": len(self.layer_weights),
        }


def create_mert_encoder(
    model_name: str = "m-a-p/MERT-v1-330M",
    freeze: bool = True,
    freeze_layer_weights: bool = False,
    unfreeze_top_n_layers: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    input_sample_rate: int = 32000,
) -> MERTEncoder:
    return MERTEncoder(
        model_name=model_name,
        freeze=freeze,
        freeze_layer_weights=freeze_layer_weights,
        unfreeze_top_n_layers=unfreeze_top_n_layers,
        device=device,
        input_sample_rate=input_sample_rate,
    )
