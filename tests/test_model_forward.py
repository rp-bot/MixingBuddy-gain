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

        # Generate and print model response
        print("\n🎯 Generating model response...")
        print("-" * 50)

        # Set model to eval mode for generation
        mm_model.eval()

        with torch.no_grad():
            # Prepare audio embeddings for generation
            anchor_features = mm_model.audio_encoder.encode(anchor_audio)
            mix_features = mm_model.audio_encoder.encode(mix_audio)

            # Project audio features
            target_dtype = mm_model.projection.projection.weight.dtype
            anchor_features = anchor_features.to(target_dtype)
            mix_features = mix_features.to(target_dtype)
            audio_embeds = mm_model.projection(anchor_features, mix_features)

            # Get text embeddings for instruction
            text_embeds = mm_model.llm.get_input_embeddings()(instr_ids)

            # Ensure same dtype
            target_embed_dtype = mm_model.llm.get_input_embeddings().weight.dtype
            if audio_embeds.dtype != target_embed_dtype:
                audio_embeds = audio_embeds.to(target_embed_dtype)
            if text_embeds.dtype != target_embed_dtype:
                text_embeds = text_embeds.to(target_embed_dtype)

            # Concatenate audio and text embeddings
            inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)

            # Build attention mask
            audio_len = audio_embeds.shape[1]
            text_len = instr_ids.shape[1]
            audio_attn = torch.ones(
                (1, audio_len), dtype=torch.long, device=inputs_embeds.device
            )
            text_attn = torch.ones(
                (1, text_len), dtype=torch.long, device=inputs_embeds.device
            )
            attention_mask = torch.cat([audio_attn, text_attn], dim=1)

            # Generate using the LLM with prepared embeddings
            generated_ids = mm_model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode the generated response (skip the input part)
            generated_text = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )

            # Also decode the original instruction and expected response for comparison
            original_instruction = tokenizer.decode(
                instr_ids[0], skip_special_tokens=True
            )
            expected_response = tokenizer.decode(resp_ids[0], skip_special_tokens=True)

            print(f"📝 Original instruction:\n{original_instruction}")
            print(f"\n🎯 Model's response:\n{generated_text}")
            print(f"\n📋 Expected response:\n{expected_response}")

        print("-" * 50)
        print("✅ Model response generation successful!")

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
