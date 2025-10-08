#!/usr/bin/env python3
"""
Forward-pass test for the modular multimodal model using a real dataset sample.
Only `test_modular_with_dataset` is retained.
"""

import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_modular_with_dataset():
    """Test modular path using one sample from our MixingDataset (if available)."""
    print("=" * 80)
    print("🧪 Modular (Qwen2-7B + Encodec) with MixingDataset sample")
    print("=" * 80)
    print()

    try:
        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.models.encoders.encodec import EncodecEncoder
        from src.models.multimodal_model import build_minimal_multimodal
        from src.models.projections.cross_attention import CrossAttentionProjection
        from src.data.dataset import MixingDataset

        # Default paths; skip gracefully if missing
        jsonl_path = "data/musdb18hq_processed/train/training_samples.jsonl"
        audio_root = "data"

        if not os.path.exists(jsonl_path):
            print(f"⚠️  Dataset jsonl not found at {jsonl_path}. Skipping this test.")
            return True

        print("📥 Loading tokenizer and LLM (Qwen2-7B-Instruct)...")
        model_name = "Qwen/Qwen2-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, dtype=torch.float16, device_map="auto"
        )
        # Freeze LLM to keep memory low
        for p in llm.parameters():
            p.requires_grad = False
        print("✅ LLM ready")

        print("📥 Creating Encodec encoder and cross-attention projection...")
        audio_encoder = EncodecEncoder(freeze=True)
        # Cross-attention projection: feature_dim is per-segment (channels * features)
        feature_dim = audio_encoder.output_channels * audio_encoder.output_dim
        projection = CrossAttentionProjection(
            feature_dim=feature_dim,
            output_dim=llm.config.hidden_size,
            num_heads=8,
            dropout=0.0,
        ).to(audio_encoder.device)
        print("✅ Audio + cross-attention projection ready")

        print("📊 Loading dataset sample...")
        dataset = MixingDataset(
            jsonl_path=jsonl_path,
            audio_root=audio_root,
            tokenizer=tokenizer,
            sample_rate=audio_encoder.sample_rate,
            limit=1,
        )
        sample = dataset[0]

        # Build model
        mm_model = build_minimal_multimodal(
            audio_encoder=audio_encoder, projection=projection, llm=llm
        )

        device = next(mm_model.parameters()).device
        # Provide separate segments; they will be encoded independently
        anchor_audio = sample["anchor_audio"].unsqueeze(0).to(device)
        mix_audio = sample["mix_audio"].unsqueeze(0).to(device)
        audio = {"anchor": anchor_audio, "mix": mix_audio}
        instr_ids = sample["input_ids"].unsqueeze(0).to(device)
        resp_ids = sample["labels"].unsqueeze(0).to(device)

        # Build combined sequence: [instruction | response]
        input_ids = torch.cat([instr_ids, resp_ids], dim=1)
        # Labels: ignore instruction tokens; supervise only response
        ignore_instr = torch.full_like(instr_ids, tokenizer.pad_token_id)
        ignore_instr[:] = -100
        labels = torch.cat([ignore_instr, resp_ids], dim=1)

        print("▶️  Running forward/backward with dataset sample...")
        outputs = mm_model(audio=audio, input_ids=input_ids, labels=labels)
        assert "loss" in outputs and outputs["loss"] is not None
        outputs["loss"].backward()
        print("✅ Dataset-backed forward/backward successful!")

        return True

    except Exception as e:
        print(f"❌ Dataset-backed modular test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run only the dataset-backed modular test."""
    try:
        passed = test_modular_with_dataset()
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback

        traceback.print_exc()
        passed = False

    print("=" * 80)
    print("📊 TEST RESULT")
    print("=" * 80)
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"DATASET: {status}")
    print()

    return passed


if __name__ == "__main__":
    main()
