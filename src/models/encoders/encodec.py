import torch
import torch.nn as nn
from typing import Optional, Union

try:
    from transformers import AutoProcessor, EncodecModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    EncodecModel = None
    AutoProcessor = None


class EncodecEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/encodec_24khz",
        freeze: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers is not available. Please install it with: pip install transformers"
            )

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = EncodecModel.from_pretrained(model_name)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.frozen = freeze
        self.model_name = model_name

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        # Add batch dimension if single sample
        # if audio.dim() == 1:
        #     audio = audio.unsqueeze(0)

        # The processor expects the audio tensor to be on the CPU.
        inputs = self.processor(
            raw_audio=audio.cpu(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )

        input_values = inputs.input_values.to(self.device)

        if self.frozen:
            with torch.no_grad():
                encoder_output = self.model.encoder(input_values)
        else:
            encoder_output = self.model.encoder(input_values)

        features = encoder_output

        # Reshape features from (batch, hidden_dim, seq_len) to (batch, seq_len, hidden_dim)
        features = features.permute(0, 2, 1)

        return features

    @property
    def output_dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def sample_rate(self) -> int:
        return self.processor.sampling_rate

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encode(audio)

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "frozen": self.frozen,
            "output_dim": self.output_dim,
            "sample_rate": self.sample_rate,
            "device": str(self.device),
        }


def create_encodec_encoder(
    model_name: str = "facebook/encodec_24khz",
    freeze: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> EncodecEncoder:
    return EncodecEncoder(
        model_name=model_name,
        freeze=freeze,
        device=device,
    )
