import torch
import torch.nn as nn
from typing import Optional, Union
import logging

from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torchaudio.transforms as T

logger = logging.getLogger(__name__)


class MERTEncoder(nn.Module):
    """MERT audio encoder that extracts features from all transformer layers.

    MERT-v1-330M outputs 25 layers of representations. This encoder:
    1. Resamples input audio from 32kHz to 24kHz (MERT's expected sample rate)
    2. Extracts all 25 hidden states and computes a learnable weighted average
    3. For audio longer than window_size_seconds (default 5s), uses a sliding window
       approach with overlap, then averages the embeddings from all windows to produce
       a single fixed-size representation (following MERT paper for global tasks)
    """

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-330M",
        freeze: bool = True,
        freeze_layer_weights: bool = False,
        unfreeze_top_n_layers: int = 0,
        unfreeze_bottom_n_layers: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        input_sample_rate: int = 32000,
        window_size_seconds: float = 5.0,
        stride_seconds: Optional[float] = None,
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
            
            # Selectively unfreeze layers if requested
            if unfreeze_top_n_layers > 0 or unfreeze_bottom_n_layers > 0:
                # MERT is based on Wav2Vec2 architecture
                # The encoder layers are in model.encoder.layers
                if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
                    total_layers = len(self.model.encoder.layers)
                    
                    # Validate no overlap between bottom and top layers
                    bottom_end = unfreeze_bottom_n_layers
                    top_start = total_layers - unfreeze_top_n_layers
                    if bottom_end > top_start:
                        logger.warning(f"Overlap detected between bottom and top unfrozen layers!")
                        logger.warning(f"  Bottom layers: 0 to {bottom_end - 1}")
                        logger.warning(f"  Top layers: {top_start} to {total_layers - 1}")
                        logger.warning(f"  Adjusting to prevent overlap...")
                        # Adjust to prevent overlap: prioritize top layers
                        max_bottom = min(unfreeze_bottom_n_layers, top_start)
                        unfreeze_bottom_n_layers = max_bottom
                        bottom_end = unfreeze_bottom_n_layers
                    
                    # Unfreeze bottom N layers (indices 0 to N-1)
                    if unfreeze_bottom_n_layers > 0:
                        bottom_layers_to_unfreeze = min(unfreeze_bottom_n_layers, total_layers)
                        for i in range(bottom_layers_to_unfreeze):
                            for param in self.model.encoder.layers[i].parameters():
                                param.requires_grad = True
                        logger.info(f"Unfroze bottom {bottom_layers_to_unfreeze} layers (layers 0 to {bottom_layers_to_unfreeze - 1}) of MERT encoder")
                    
                    # Unfreeze top N layers (higher indices = later/top layers)
                    if unfreeze_top_n_layers > 0:
                        top_layers_to_unfreeze = min(unfreeze_top_n_layers, total_layers)
                        for i in range(total_layers - top_layers_to_unfreeze, total_layers):
                            for param in self.model.encoder.layers[i].parameters():
                                param.requires_grad = True
                        logger.info(f"Unfroze top {top_layers_to_unfreeze} layers (layers {total_layers - top_layers_to_unfreeze} to {total_layers - 1}) of MERT encoder")
                    
                    # Set model to train mode for the unfrozen layers
                    self.model.train()
                else:
                    logger.warning(f"Could not find encoder layers to unfreeze. Model structure: {type(self.model)}")
        else:
            # Enable gradient checkpointing only if model is trainable
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()

        self.frozen = freeze
        self.unfreeze_top_n_layers = unfreeze_top_n_layers
        self.unfreeze_bottom_n_layers = unfreeze_bottom_n_layers
        self.model_name = model_name
        self.input_sample_rate = input_sample_rate
        self.window_size_seconds = window_size_seconds
        # Default stride is half the window size for 50% overlap (as in MERT paper)
        self.stride_seconds = stride_seconds if stride_seconds is not None else window_size_seconds / 2.0

        # Create resampler if needed
        mert_sample_rate = self.processor.sampling_rate
        if input_sample_rate != mert_sample_rate:
            self.resampler = T.Resample(
                orig_freq=input_sample_rate, new_freq=mert_sample_rate
            )
            self.resampler = self.resampler.to(self.device)
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

        For audio longer than window_size_seconds (default 5s), the audio is processed using
        a sliding window approach with overlap. The embeddings from all windows are averaged
        to produce a single fixed-size representation, following the MERT paper's approach
        for global tasks.

        Args:
            audio: Input audio tensor, shape [batch_size, num_samples] or [num_samples]
                   Expected to be at self.input_sample_rate

        Returns:
            features: Weighted average of layer features, shape [batch, time_steps, 1024]
                     For audio longer than window_size_seconds, this is the averaged embedding
                     across all sliding windows, maintaining the same time_steps dimension.
        """
        # Handle single sample case
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Ensure audio is on the correct device
        audio = audio.to(self.device)

        # Resample if needed
        if self.needs_resampling:
            audio = self.resampler(audio)

        # After resampling, audio is at MERT's native sample rate (24kHz)
        mert_sample_rate = self.processor.sampling_rate
        window_samples = int(self.window_size_seconds * mert_sample_rate)
        stride_samples = int(self.stride_seconds * mert_sample_rate)

        batch_size = audio.shape[0]
        
        # Optimize: Check if all samples are short enough to batch process
        # Get the maximum length across all samples in the batch
        if audio.dim() == 2:
            max_samples = audio.shape[1]
        else:
            # Shouldn't happen after unsqueeze, but handle it
            max_samples = max(audio[i].shape[0] for i in range(batch_size)) if batch_size > 0 else 0
        
        if max_samples <= window_samples:
            # All samples fit in one window - batch process them all at once
            # Audio is already 2D [batch, samples] after resampling
            # Process entire batch at once
            features = self._encode_single_chunk(audio)  # [batch, time_steps, 1024]
            return features
        
        # Some samples are too long - need sliding window or variable length handling
        # Process each sample separately (unavoidable due to variable lengths and sliding windows)
        all_features = []

        for i in range(batch_size):
            sample_audio = audio[i]  # [num_samples]
            num_samples = sample_audio.shape[0]

            # Check if we need to use sliding window
            if num_samples > window_samples:
                # Use sliding window approach with overlap
                # Optimize: Batch all chunks together when possible
                chunk_embeddings = []
                chunks = []
                start_idx = 0

                while start_idx < num_samples:
                    end_idx = min(start_idx + window_samples, num_samples)
                    chunk = sample_audio[start_idx:end_idx]

                    # If the last chunk is shorter than window_samples, pad it
                    if chunk.shape[0] < window_samples:
                        padding = torch.zeros(
                            window_samples - chunk.shape[0],
                            device=chunk.device,
                            dtype=chunk.dtype
                        )
                        chunk = torch.cat([chunk, padding], dim=0)
                    
                    chunks.append(chunk)
                    start_idx += stride_samples
                    
                    # If we've processed the last possible window, break
                    if end_idx >= num_samples:
                        break

                # Batch process all chunks at once for better GPU utilization
                if len(chunks) > 1:
                    chunks_batched = torch.stack(chunks, dim=0)  # [num_chunks, window_samples]
                    chunk_feats = self._encode_single_chunk(chunks_batched)  # [num_chunks, time_steps, 1024]
                    # Mean pool over time dimension for each chunk
                    chunk_embeddings = chunk_feats.mean(dim=1)  # [num_chunks, 1024]
                else:
                    # Single chunk
                    chunk_feat = self._encode_single_chunk(chunks[0].unsqueeze(0))  # [1, time_steps, 1024]
                    chunk_embeddings = [chunk_feat.mean(dim=1).squeeze(0)]  # [1024]

                # Average all chunk embeddings to get a single fixed-size representation
                if isinstance(chunk_embeddings, list):
                    chunk_embeddings = torch.stack(chunk_embeddings, dim=0)  # [num_chunks, 1024]
                # Average across chunks: [1024]
                averaged_embedding = chunk_embeddings.mean(dim=0)  # [1024]
                
                # Expand to match expected output shape [time_steps=1, 1024]
                # This maintains compatibility with downstream models expecting [batch, time_steps, 1024]
                features = averaged_embedding.unsqueeze(0)  # [1, 1024]
            else:
                # Process normally (single chunk)
                features = self._encode_single_chunk(sample_audio.unsqueeze(0))  # [batch=1, time_steps, 1024]
                features = features.squeeze(0)  # [time_steps, 1024]

            all_features.append(features)

        # Stack all batch samples
        # For averaged embeddings (long audio), features is [1, 1024]
        # For normal processing (short audio), features is [time_steps, 1024]
        # We need to handle both cases
        max_time_steps = max(feat.shape[0] for feat in all_features)
        feature_dim = all_features[0].shape[1]
        
        # Pad and stack efficiently
        padded_features = []
        for feat in all_features:
            if feat.shape[0] < max_time_steps:
                padding = torch.zeros(
                    max_time_steps - feat.shape[0],
                    feature_dim,
                    device=feat.device,
                    dtype=feat.dtype
                )
                feat = torch.cat([feat, padding], dim=0)
            padded_features.append(feat)
        
        return torch.stack(padded_features, dim=0)  # [batch, max_time_steps, 1024]

    def _encode_single_chunk(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode a single audio chunk (assumed to be <= window_size_seconds).

        Args:
            audio: Input audio tensor, shape [batch_size, num_samples] or [num_samples]
                   Expected to be at MERT's native sample rate (24kHz)

        Returns:
            features: Weighted average of layer features, shape [batch, time_steps, 1024]
        """
        # Handle single sample case
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Optimize: Keep audio on GPU and convert to numpy only when necessary
        # The processor needs numpy, but we can minimize transfers
        if isinstance(audio, torch.Tensor):
            # Convert to numpy efficiently - only move to CPU if not already there
            if audio.device.type == "cpu":
                audio_cpu = audio
            else:
                audio_cpu = audio.cpu()
            
            # Convert to numpy - handle both 1D and 2D
            if audio_cpu.dim() == 1:
                audio_np = audio_cpu.numpy()
            else:
                # For batched input, convert to list of numpy arrays
                # This is necessary for the processor, but we batch process later
                audio_np = [audio_cpu[i].numpy() for i in range(audio_cpu.shape[0])]
        else:
            audio_np = audio
        
        inputs = self.processor(
            raw_speech=audio_np,
            sampling_rate=self.processor.sampling_rate,
            return_tensors="pt",
        )

        # Move inputs to model device and ensure correct shape
        input_values = inputs.input_values.to(self.device)

        # Ensure input_values is 2D: [batch_size, sequence_length]
        # Handle any extra dimensions that might have been added
        original_shape = input_values.shape
        while input_values.dim() > 2:
            # Remove singleton dimensions
            if input_values.shape[0] == 1:
                input_values = input_values.squeeze(0)
            elif input_values.shape[1] == 1:
                input_values = input_values.squeeze(1)
            else:
                # If no singleton dims, flatten extra dimensions
                # Keep first dim as batch, flatten the rest
                batch_size = input_values.shape[0]
                input_values = input_values.view(batch_size, -1)
        
        # Final safety check
        if input_values.dim() != 2:
            raise ValueError(
                f"Expected 2D input_values after processing, but got shape {input_values.shape} "
                f"(original shape: {original_shape}). Audio input shape was {audio.shape if isinstance(audio, torch.Tensor) else type(audio)}"
            )

        # Extract features with all hidden states
        # Don't use no_grad if we have unfrozen layers (top or bottom)
        if self.frozen and self.unfreeze_top_n_layers == 0 and self.unfreeze_bottom_n_layers == 0:
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

        # Optimize: Avoid stacking all layers if we only need the last one
        if self.freeze_layer_weights:
            # Use the last layer directly: [batch, time_steps, 1024]
            features = outputs.hidden_states[-1]
            return features

        # For weighted average, compute incrementally to avoid stacking all layers
        # This saves memory and can be faster than stacking then summing
        # outputs.hidden_states is a tuple of (num_layers) tensors
        # Each tensor has shape [batch, time_steps, 1024]
        
        # Normalize layer weights using softmax to ensure they sum to 1
        normalized_weights = torch.softmax(self.layer_weights, dim=0)
        
        # Compute weighted sum incrementally instead of stacking all layers
        # This avoids creating a large [25, batch, time_steps, 1024] tensor
        features = None
        for i, hidden_state in enumerate(outputs.hidden_states):
            weight = normalized_weights[i]
            weighted_state = hidden_state * weight
            if features is None:
                features = weighted_state
            else:
                features = features + weighted_state

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
            "unfreeze_bottom_n_layers": self.unfreeze_bottom_n_layers,
            "output_dim": self.output_dim,
            "sample_rate": self.sample_rate,
            "input_sample_rate": self.input_sample_rate,
            "needs_resampling": self.needs_resampling,
            "device": str(self.device),
            "layer_aggregation": "learnable_weighted_average",
            "num_layer_weights": len(self.layer_weights),
            "window_size_seconds": self.window_size_seconds,
            "stride_seconds": self.stride_seconds,
        }


def create_mert_encoder(
    model_name: str = "m-a-p/MERT-v1-330M",
    freeze: bool = True,
    freeze_layer_weights: bool = False,
    unfreeze_top_n_layers: int = 0,
    unfreeze_bottom_n_layers: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    input_sample_rate: int = 32000,
    window_size_seconds: float = 5.0,
    stride_seconds: Optional[float] = None,
) -> MERTEncoder:
    return MERTEncoder(
        model_name=model_name,
        freeze=freeze,
        freeze_layer_weights=freeze_layer_weights,
        unfreeze_top_n_layers=unfreeze_top_n_layers,
        unfreeze_bottom_n_layers=unfreeze_bottom_n_layers,
        device=device,
        input_sample_rate=input_sample_rate,
        window_size_seconds=window_size_seconds,
        stride_seconds=stride_seconds,
    )
