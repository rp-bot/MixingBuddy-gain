"""
Data collator for stem classification and gain regression.
"""

from typing import Dict, List, Any
import torch


class StemGainDataCollator:
    """Collator for batching audio and labels.
    
    Handles padding of variable-length audio sequences.
    """
    
    def __init__(self, pad_value: float = 0.0):
        """
        Args:
            pad_value: Value to use for padding audio sequences
        """
        self.pad_value = pad_value
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            features: List of samples from dataset
        
        Returns:
            Batched dictionary with:
                - audio: [batch_size, max_length] padded audio
                - stem_label: [batch_size] classification labels
                - gain_label: [batch_size] regression labels
                - attention_mask: [batch_size, max_length] mask for padding
        """
        # Extract audio sequences
        audio_list = [f["audio"] for f in features]
        
        # Ensure all audio tensors are 1D and float32
        audio_list = [
            audio.clone().float() if isinstance(audio, torch.Tensor) else torch.tensor(audio, dtype=torch.float32)
            for audio in audio_list
        ]
        
        # Flatten to 1D if needed (librosa should already give 1D, but safety check)
        audio_list = [audio.squeeze() if audio.dim() > 1 else audio for audio in audio_list]
        
        # Find maximum length
        max_length = max(audio.shape[-1] if audio.dim() > 0 else len(audio) for audio in audio_list)
        
        # Pad each audio tensor to max_length
        padded_audio_list = []
        attention_mask_list = []
        for audio in audio_list:
            current_length = audio.shape[-1] if audio.dim() > 0 else len(audio)
            if current_length < max_length:
                # Pad with zeros
                padding_size = max_length - current_length
                padded = torch.nn.functional.pad(
                    audio, (0, padding_size), mode="constant", value=self.pad_value
                )
                # Create mask: 1 for real audio, 0 for padding
                mask = torch.cat([
                    torch.ones(current_length, dtype=torch.float32),
                    torch.zeros(padding_size, dtype=torch.float32)
                ])
            else:
                padded = audio[:max_length]  # Crop if too long
                mask = torch.ones(max_length, dtype=torch.float32)
            
            padded_audio_list.append(padded)
            attention_mask_list.append(mask)
        
        # Stack to create batch: [batch_size, max_length]
        padded_audio = torch.stack(padded_audio_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)
        
        # Extract labels
        stem_labels = torch.tensor([f["stem_label"] for f in features], dtype=torch.long)
        gain_labels = torch.tensor([f["gain_label"] for f in features], dtype=torch.float32)
        
        # Extract metadata (optional, for logging/debugging)
        global_uids = [f["global_uid"] for f in features]
        target_stems = [f["target_stem"] for f in features]
        
        return {
            "audio": padded_audio,
            "stem_label": stem_labels,
            "gain_label": gain_labels,
            "attention_mask": attention_mask,
            "global_uid": global_uids,
            "target_stem": target_stems,
        }

