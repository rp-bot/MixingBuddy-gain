#!/usr/bin/env python3
"""
Test script for EncodecEncoder - Phase 3 verification.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from src.models.encoders.encodec import EncodecEncoder, create_encodec_encoder


def test_encodec_basic():
    """Test basic Encodec functionality."""
    print("🧪 Testing EncodecEncoder...")

    try:
        # Create encoder
        print("📥 Creating Encodec encoder...")
        encoder = EncodecEncoder(freeze=True)
        print("✅ Encoder created successfully")

        # Test with random audio
        print("🔍 Testing with random audio...")
        batch_size = 2
        duration_sec = 10
        sample_rate = encoder.sample_rate
        audio_length = duration_sec * sample_rate

        # Create random audio
        audio = torch.randn(batch_size, audio_length)
        print(f"  Input shape: {audio.shape}")

        # Encode
        features = encoder.encode(audio)
        print(f"  Output shape: {features.shape}")
        print(f"  Output dim: {encoder.output_dim}")

        # Verify shapes - should be (batch, time, channels, features)
        assert features.shape[0] == batch_size, (
            f"Expected batch size {batch_size}, got {features.shape[0]}"
        )
        assert features.shape[2] == encoder.output_channels, (
            f"Expected channels {encoder.output_channels}, got {features.shape[2]}"
        )
        assert features.shape[3] == encoder.output_dim, (
            f"Expected feature dim {encoder.output_dim}, got {features.shape[3]}"
        )
        print("✅ Output shape is correct")

        # Test single sample
        print("🔍 Testing single sample...")
        single_audio = torch.randn(audio_length)
        single_features = encoder.encode(single_audio)
        print(f"  Single input shape: {single_audio.shape}")
        print(f"  Single output shape: {single_features.shape}")

        # Verify single sample shape
        assert single_features.shape[0] == 1, (
            f"Expected batch size 1, got {single_features.shape[0]}"
        )
        assert single_features.shape[2] == encoder.output_channels, (
            f"Expected channels {encoder.output_channels}, got {single_features.shape[2]}"
        )
        assert single_features.shape[3] == encoder.output_dim, (
            f"Expected feature dim {encoder.output_dim}, got {single_features.shape[3]}"
        )
        print("✅ Single sample shape is correct")

        # Test frozen parameters
        print("🔍 Testing frozen parameters...")
        frozen_params = sum(1 for p in encoder.parameters() if not p.requires_grad)
        total_params = sum(1 for p in encoder.parameters())
        print(f"  Frozen parameters: {frozen_params}/{total_params}")
        assert frozen_params == total_params, "All parameters should be frozen"
        print("✅ All parameters are frozen")

        # Test model info
        print("🔍 Testing model info...")
        info = encoder.get_model_info()
        print(f"  Model info: {info}")
        assert info["frozen"] == True
        assert info["output_dim"] == 75
        assert info["output_channels"] == 8
        assert info["sample_rate"] == 24000
        print("✅ Model info is correct")

        print("✅ All basic tests passed!")
        return True

    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_encodec_with_real_audio():
    """Test Encodec with real audio from our dataset."""
    print("\n🔍 Testing with real dataset audio...")

    try:
        # Import dataset
        from transformers import AutoTokenizer
        from src.data.dataset import MixingDataset

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create dataset
        dataset = MixingDataset(
            jsonl_path="data/musdb18hq_processed/train/training_samples.jsonl",
            audio_root="data",
            tokenizer=tokenizer,
            limit=3,  # Just test with 3 samples
        )

        # Create encoder
        encoder = EncodecEncoder(freeze=True)

        # Test with dataset samples
        for i in range(min(2, len(dataset))):
            print(f"  Testing sample {i}...")
            sample = dataset[i]
            audio = sample["audio"]  # Shape: (samples,)

            # Resample to 24kHz if needed (our dataset is 48kHz)
            if len(audio) > 0:
                # Simple downsampling by taking every other sample
                audio_24k = audio[::2]  # Downsample from 48kHz to 24kHz

                # Encode
                features = encoder.encode(audio_24k)
                print(
                    f"    Audio length: {len(audio_24k)} samples ({len(audio_24k) / 24000:.2f}s)"
                )
                print(f"    Features shape: {features.shape}")

                # Verify features are reasonable
                assert not torch.isnan(features).any(), "Features contain NaN"
                assert not torch.isinf(features).any(), "Features contain Inf"
                print(
                    f"    Feature range: [{features.min():.3f}, {features.max():.3f}]"
                )
                print("    ✅ Sample processed successfully")

        print("✅ Real audio tests passed!")
        return True

    except Exception as e:
        print(f"❌ Real audio test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_encodec_factory():
    """Test the factory function."""
    print("\n🔍 Testing factory function...")

    try:
        # Test factory function
        encoder = create_encodec_encoder(freeze=True)
        assert isinstance(encoder, EncodecEncoder)
        print("✅ Factory function works")

        # Test different parameters
        encoder2 = create_encodec_encoder(
            target_bandwidth=3.0, freeze=False, device="cpu"
        )
        assert encoder2.target_bandwidth == 3.0
        assert encoder2.frozen == False
        print("✅ Factory function with custom parameters works")

        return True

    except Exception as e:
        print(f"❌ Factory test failed: {e}")
        return False


def test_encodec_performance():
    """Test encoding performance."""
    print("\n🔍 Testing encoding performance...")

    try:
        import time

        encoder = EncodecEncoder(freeze=True)

        # Test with different audio lengths
        durations = [1, 5, 10]  # seconds
        sample_rate = encoder.sample_rate

        for duration in durations:
            audio_length = duration * sample_rate
            audio = torch.randn(1, audio_length)

            # Time the encoding
            start_time = time.time()
            features = encoder.encode(audio)
            end_time = time.time()

            encoding_time = end_time - start_time
            real_time_factor = encoding_time / duration

            print(
                f"  {duration}s audio: {encoding_time:.3f}s encoding time (RTF: {real_time_factor:.2f})"
            )
            print(f"    Features shape: {features.shape}")

        print("✅ Performance test completed")
        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Phase 3 Encodec Tests\n")

    success1 = test_encodec_basic()
    success2 = test_encodec_factory()
    success3 = test_encodec_performance()
    success4 = test_encodec_with_real_audio()

    if success1 and success2 and success3 and success4:
        print("\n🎉 All Phase 3 tests passed! Encodec encoder is ready.")
    else:
        print("\n💥 Some tests failed. Please check the errors above.")
        sys.exit(1)
