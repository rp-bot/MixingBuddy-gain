#!/usr/bin/env python3
"""
Test script for MixingDataset - Phase 2 verification.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer
from src.data.dataset import MixingDataset


def test_dataset_loading():
    """Test basic dataset loading functionality."""
    print("🧪 Testing MixingDataset...")

    # Paths
    jsonl_path = "data/musdb18hq_processed/train/training_samples.jsonl"
    audio_root = "data"

    # Check if files exist
    if not Path(jsonl_path).exists():
        print(f"❌ JSONL file not found: {jsonl_path}")
        return False

    if not Path(audio_root).exists():
        print(f"❌ Audio root not found: {audio_root}")
        return False

    try:
        # Load tokenizer
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded")

        # Create dataset (limit to 5 samples for testing)
        print("📊 Creating dataset...")
        dataset = MixingDataset(
            jsonl_path=jsonl_path,
            audio_root=audio_root,
            tokenizer=tokenizer,
            limit=5,  # Just test with 5 samples
        )
        print(f"✅ Dataset created with {len(dataset)} samples")

        # Test loading one sample
        print("🔍 Testing sample loading...")
        sample = dataset[0]

        print(f"  Anchor shape: {sample['anchor_audio'].shape}")
        print(f"  Mix shape: {sample['mix_audio'].shape}")
        print(f"  Silence samples: {sample['silence_samples']}")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")
        print(f"  Instruction: {sample['instruction'][:100]}...")
        print(f"  Response: {sample['response'][:100]}...")
        print(f"  Anchor stem: {sample['anchor_stem']}")
        print(f"  Target stem: {sample['target_stem']}")
        print(f"  Error category: {sample['error_category']}")

        # Verify audio properties
        # Verify per-part properties
        anchor = sample["anchor_audio"]
        mix = sample["mix_audio"]
        sr = sample["sample_rate"]
        expected_anchor = int(10 * sr)
        expected_mix = int(10 * sr)
        if abs(len(anchor) - expected_anchor) > int(0.01 * expected_anchor):
            print(f"⚠️  Anchor length unexpected: {len(anchor)} vs ~{expected_anchor}")
        if abs(len(mix) - expected_mix) > int(0.01 * expected_mix):
            print(f"⚠️  Mix length unexpected: {len(mix)} vs ~{expected_mix}")
        else:
            print("✅ Anchor/Mix lengths look correct")

        # Test tokenization
        instruction_text = tokenizer.decode(
            sample["input_ids"], skip_special_tokens=True
        )
        response_text = tokenizer.decode(sample["labels"], skip_special_tokens=True)

        print(f"  Decoded instruction: {instruction_text[:100]}...")
        print(f"  Decoded response: {response_text[:100]}...")

        # Test loading multiple samples
        print("🔄 Testing batch loading...")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            assert sample["anchor_audio"].shape[0] > 0, f"Sample {i} has empty anchor"
            assert sample["mix_audio"].shape[0] > 0, f"Sample {i} has empty mix"
            assert sample["input_ids"].shape[0] > 0, f"Sample {i} has empty input_ids"
            assert sample["labels"].shape[0] > 0, f"Sample {i} has empty labels"
            print(f"  Sample {i}: ✅ OK")

        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sample_info():
    """Test getting sample metadata without loading audio."""
    print("\n🔍 Testing sample info...")

    jsonl_path = "data/musdb18hq_processed/train/training_samples.jsonl"
    audio_root = "data"

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        dataset = MixingDataset(
            jsonl_path=jsonl_path,
            audio_root=audio_root,
            tokenizer=tokenizer,
            limit=3,
        )

        # Test getting sample info
        for i in range(len(dataset)):
            info = dataset.get_sample_info(i)
            print(f"  Sample {i}:")
            print(f"    Global UID: {info['global_uid']}")
            print(f"    Anchor: {info['meta']['anchor_stem']}")
            print(f"    Target: {info['meta']['target_stem']}")
            print(f"    Error: {info['meta']['error_category']}")
            print(
                f"    Time: {info['meta']['time_ref']['start_sec']}-{info['meta']['time_ref']['end_sec']}s"
            )

        print("✅ Sample info test passed!")
        return True

    except Exception as e:
        print(f"❌ Sample info test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Phase 2 Dataset Tests\n")

    success1 = test_dataset_loading()
    success2 = test_sample_info()

    if success1 and success2:
        print("\n🎉 All Phase 2 tests passed! Dataset is ready for training.")
    else:
        print("\n💥 Some tests failed. Please check the errors above.")
        sys.exit(1)
