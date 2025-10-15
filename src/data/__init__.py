"""
Data processing and management modules for LLM fine-tuning.
"""

from src.data.collator import MultimodalDataCollator
from src.data.dataset import MixingDataset
from src.data.audio_processing import load_track_stems, chunk_audio
from src.data.gating import compute_rms_dbfs, apply_gating
from src.data.error_injection import sample_error_category, apply_gain_error
from src.data.text_generation import create_instruction, create_response
from src.data.synthesis import synthesize_chunk, process_split

__all__ = [
    "MultimodalDataCollator",
    "MixingDataset",
    "load_track_stems",
    "chunk_audio",
    "compute_rms_dbfs",
    "apply_gating",
    "sample_error_category",
    "apply_gain_error",
    "create_instruction",
    "create_response",
    "synthesize_chunk",
    "process_split",
]
