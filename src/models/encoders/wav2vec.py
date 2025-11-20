import torch
import torch.nn as nn
from typing import Optional, Union
import logging
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torchaudio.transforms as T

logger = logging.getLogger(__name__)

class Wav2VecEncoder(nn.Module):
    """Wav2Vec 2.0 audio encoder.
    
    Supports extracting features from either:
    1. The CNN feature extractor (feature_type="cnn") -> [batch, time, 512]
    2. The full Transformer (feature_type="transformer") -> [batch, time, 768]
    
    Input audio is automatically resampled to 16kHz.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        freeze: bool = True,
        feature_type: str = "cnn",  # "cnn" or "transformer"
        device: Optional[Union[str, torch.device]] = None,
        input_sample_rate: int = 32000,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.feature_type = feature_type.lower()
        self.input_sample_rate = input_sample_rate
        self.frozen = freeze
        
        if self.feature_type not in ["cnn", "transformer"]:
            raise ValueError(f"Invalid feature_type: {feature_type}. Must be 'cnn' or 'transformer'")
            
        logger.info(f"Loading Wav2Vec2 model: {model_name} (feature_type={feature_type})")
        
        # Load model and processor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Resampling setup
        wav2vec_sample_rate = 16000  # Wav2Vec2 always uses 16kHz
        if input_sample_rate != wav2vec_sample_rate:
            self.resampler = T.Resample(
                orig_freq=input_sample_rate, new_freq=wav2vec_sample_rate
            ).to(self.device)
            self.needs_resampling = True
        else:
            self.resampler = None
            self.needs_resampling = False
            
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to features.
        
        Args:
            audio: Input audio tensor [batch, samples] or [samples]
            
        Returns:
            features: [batch, time, feature_dim]
        """
        # Handle single sample
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        audio = audio.to(self.device)
        
        # Resample
        if self.needs_resampling:
            audio = self.resampler(audio)
            
        # Process input
        # Wav2Vec2 expects normalized audio. The processor does this, but accepts numpy.
        # To stay in torch/GPU, we manually normalize if needed, or trust the model handles it.
        # Wav2Vec2Model expects raw waveform input_values.
        # Best practice with HF models is to use the processor, but that requires CPU roundtrip.
        # For efficiency, we'll feed tensor directly. Wav2Vec2 expects zero mean / unit var per sample usually.
        # Let's implement simple normalization like the processor does.
        
        with torch.no_grad():
            # Normalize: (x - mean) / sqrt(var + 1e-7)
            if self.frozen: # Only do this carefully if we are not training e2e
                 mean = audio.mean(dim=-1, keepdim=True)
                 var = audio.var(dim=-1, keepdim=True)
                 audio = (audio - mean) / torch.sqrt(var + 1e-7)

        # Forward pass
        if self.frozen:
            with torch.no_grad():
                if self.feature_type == "cnn":
                    # Extract CNN features
                    # model.feature_extractor returns raw features
                    features = self.model.feature_extractor(audio)
                    # Shape: [batch, dim, time] -> need [batch, time, dim]
                    features = features.transpose(1, 2)
                else:
                    # Transformer output
                    outputs = self.model(audio)
                    features = outputs.last_hidden_state
        else:
            if self.feature_type == "cnn":
                features = self.model.feature_extractor(audio)
                features = features.transpose(1, 2)
            else:
                outputs = self.model(audio)
                features = outputs.last_hidden_state
                
        return features

    @property
    def output_dim(self) -> int:
        if self.feature_type == "cnn":
            # Check config for feature extractor dim, usually 512 for base
            return self.model.config.conv_dim[-1]
        else:
            return self.model.config.hidden_size
            
    @property
    def sample_rate(self) -> int:
        return 16000
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encode(audio)

