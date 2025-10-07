#!/usr/bin/env python3
"""
Test script for LinearProjection - Phase 4 verification.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.projections.linear import LinearProjection, create_linear_projection


def test_projection_basic():
    """Test basic projection functionality."""
    print("🧪 Testing LinearProjection...")

    try:
        # Test with Encodec-like input
        input_dim = 8 * 75  # 600 (channels * features)
        output_dim = 3584  # Qwen2-7B embedding dimension

        print("📥 Creating projection layer...")
        projection = LinearProjection(
            input_dim=input_dim, output_dim=output_dim, dropout=0.1, activation="relu"
        )
        print("✅ Projection layer created successfully")

        # Test with 4D input (from Encodec)
        print("🔍 Testing with 4D input (Encodec format)...")
        batch_size = 2
        time_steps = 10
        channels = 8
        features = 75

        # Create input tensor: (batch, time, channels, features)
        audio_features = torch.randn(batch_size, time_steps, channels, features)
        print(f"  Input shape: {audio_features.shape}")

        # Project
        projected = projection(audio_features)
        print(f"  Output shape: {projected.shape}")

        # Verify output shape
        expected_shape = (batch_size, time_steps, output_dim)
        assert projected.shape == expected_shape, (
            f"Expected {expected_shape}, got {projected.shape}"
        )
        print("✅ 4D input projection works correctly")

        # Test with 3D input (already flattened)
        print("🔍 Testing with 3D input (flattened format)...")
        flattened_input = torch.randn(batch_size, time_steps, input_dim)
        print(f"  Input shape: {flattened_input.shape}")

        projected_3d = projection(flattened_input)
        print(f"  Output shape: {projected_3d.shape}")

        # Verify output shape
        assert projected_3d.shape == expected_shape, (
            f"Expected {expected_shape}, got {projected_3d.shape}"
        )
        print("✅ 3D input projection works correctly")

        # Test model info
        print("🔍 Testing model info...")
        info = projection.get_model_info()
        print(f"  Model info: {info}")
        assert info["input_dim"] == input_dim
        assert info["output_dim"] == output_dim
        assert info["dropout"] == 0.1
        assert info["activation"] == "ReLU()"
        print("✅ Model info is correct")

        print("✅ All basic tests passed!")
        return True

    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_projection_with_encodec():
    """Test projection with actual Encodec features."""
    print("\n🔍 Testing projection with Encodec features...")

    try:
        from src.models.encoders.encodec import EncodecEncoder

        # Create encoder and projection
        encoder = EncodecEncoder(freeze=True)
        projection = LinearProjection(
            input_dim=encoder.output_channels * encoder.output_dim,  # 8 * 75 = 600
            output_dim=3584,  # Qwen2-7B embedding dimension
            dropout=0.1,
            activation="relu",
        )

        # Move projection to same device as encoder
        projection = projection.to(encoder.device)

        # Create test audio
        audio = torch.randn(1, 24000)  # 1 second of audio
        print(f"  Audio shape: {audio.shape}")

        # Encode audio
        features = encoder.encode(audio)
        print(f"  Encodec features shape: {features.shape}")
        print(f"  Encodec features dtype: {features.dtype}")

        # Ensure features are float32
        features = features.float()

        # Project features
        projected = projection(features)
        print(f"  Projected features shape: {projected.shape}")

        # Verify shapes
        expected_projected_shape = (1, features.shape[1], 3584)
        assert projected.shape == expected_projected_shape, (
            f"Expected {expected_projected_shape}, got {projected.shape}"
        )
        print("✅ Encodec + projection integration works")

        # Test with longer audio
        print("🔍 Testing with longer audio...")
        long_audio = torch.randn(1, 240000)  # 10 seconds
        long_features = encoder.encode(long_audio)
        long_features = long_features.float()  # Ensure float dtype
        long_projected = projection(long_features)
        print(f"  Long audio projected shape: {long_projected.shape}")

        expected_long_shape = (1, long_features.shape[1], 3584)
        assert long_projected.shape == expected_long_shape, (
            f"Expected {expected_long_shape}, got {long_projected.shape}"
        )
        print("✅ Long audio projection works")

        return True

    except Exception as e:
        print(f"❌ Encodec integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_projection_variants():
    """Test different projection configurations."""
    print("\n🔍 Testing projection variants...")

    try:
        input_dim = 600
        output_dim = 3584

        # Test without activation
        proj1 = LinearProjection(input_dim, output_dim, activation=None)
        x = torch.randn(2, 10, input_dim)
        y1 = proj1(x)
        assert y1.shape == (2, 10, output_dim)
        print("✅ No activation works")

        # Test with GELU activation
        proj2 = LinearProjection(input_dim, output_dim, activation="gelu")
        y2 = proj2(x)
        assert y2.shape == (2, 10, output_dim)
        print("✅ GELU activation works")

        # Test with dropout
        proj3 = LinearProjection(input_dim, output_dim, dropout=0.5)
        y3 = proj3(x)
        assert y3.shape == (2, 10, output_dim)
        print("✅ Dropout works")

        # Test factory function
        proj4 = create_linear_projection(input_dim, output_dim, activation="tanh")
        y4 = proj4(x)
        assert y4.shape == (2, 10, output_dim)
        print("✅ Factory function works")

        return True

    except Exception as e:
        print(f"❌ Variants test failed: {e}")
        return False


def test_projection_gradients():
    """Test that projection parameters are trainable."""
    print("\n🔍 Testing gradient flow...")

    try:
        projection = LinearProjection(600, 3584)

        # Create dummy input and target
        x = torch.randn(2, 10, 600, requires_grad=True)
        target = torch.randn(2, 10, 3584)

        # Forward pass
        output = projection(x)
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check gradients
        assert projection.projection.weight.grad is not None
        assert projection.projection.bias.grad is not None
        print("✅ Gradients flow correctly")

        # Check parameter counts
        total_params = sum(p.numel() for p in projection.parameters())
        expected_params = 600 * 3584 + 3584  # weights + bias
        assert total_params == expected_params, (
            f"Expected {expected_params} params, got {total_params}"
        )
        print(f"✅ Parameter count correct: {total_params}")

        return True

    except Exception as e:
        print(f"❌ Gradient test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Phase 4 Projection Tests\n")

    success1 = test_projection_basic()
    success2 = test_projection_with_encodec()
    success3 = test_projection_variants()
    success4 = test_projection_gradients()

    if success1 and success2 and success3 and success4:
        print("\n🎉 All Phase 4 tests passed! Projection layer is ready.")
    else:
        print("\n💥 Some tests failed. Please check the errors above.")
        sys.exit(1)
