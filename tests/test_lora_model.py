"""
Tests for LoRA model functionality.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.models.lora_model import LoRAModel, create_lora_model


class TestLoRAModel:
    """Test cases for LoRAModel class."""

    def test_init(self, sample_config):
        """Test LoRAModel initialization."""
        model = LoRAModel(sample_config)

        assert model.config == sample_config
        assert model.model is None
        assert model.tokenizer is None
        assert model.peft_model is None

    @patch("src.models.lora_model.AutoModelForCausalLM")
    def test_load_model_basic(self, mock_model_class, sample_config):
        """Test loading basic model without quantization."""
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        model = LoRAModel(sample_config)
        loaded_model = model.load_model()

        assert loaded_model == mock_model
        assert model.model == mock_model
        mock_model_class.from_pretrained.assert_called_once()

    @patch("src.models.lora_model.AutoModelForCausalLM")
    @patch("src.models.lora_model.BitsAndBytesConfig")
    @patch("src.models.lora_model.prepare_model_for_kbit_training")
    def test_load_model_with_4bit_quantization(
        self, mock_prepare, mock_bnb_config, mock_model_class, sample_config
    ):
        """Test loading model with 4-bit quantization."""
        # Add quantization config
        sample_config.quantization = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        }

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_prepare.return_value = mock_model

        model = LoRAModel(sample_config)
        loaded_model = model.load_model()

        assert loaded_model == mock_model
        assert model.model == mock_model
        mock_bnb_config.assert_called_once()
        mock_prepare.assert_called_once_with(mock_model)

    @patch("src.models.lora_model.AutoModelForCausalLM")
    @patch("src.models.lora_model.BitsAndBytesConfig")
    def test_load_model_with_8bit_quantization(
        self, mock_bnb_config, mock_model_class, sample_config
    ):
        """Test loading model with 8-bit quantization."""
        # Add quantization config
        sample_config.quantization = {
            "load_in_8bit": True,
        }

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        model = LoRAModel(sample_config)
        loaded_model = model.load_model()

        assert loaded_model == mock_model
        assert model.model == mock_model
        mock_bnb_config.assert_called_once()

    @patch("src.models.lora_model.Qwen2AudioForConditionalGeneration")
    def test_load_model_qwen2_audio(self, mock_qwen_class, sample_config):
        """Test loading Qwen2-Audio model."""
        sample_config.pretrained_model_name_or_path = "Qwen/Qwen2-Audio-7B-Instruct"

        mock_model = Mock()
        mock_qwen_class.from_pretrained.return_value = mock_model

        model = LoRAModel(sample_config)
        loaded_model = model.load_model()

        assert loaded_model == mock_model
        assert model.model == mock_model
        mock_qwen_class.from_pretrained.assert_called_once()

    @patch("src.models.lora_model.AutoTokenizer")
    def test_load_tokenizer(self, mock_tokenizer_class, sample_config):
        """Test loading tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        model = LoRAModel(sample_config)
        loaded_tokenizer = model.load_tokenizer()

        assert loaded_tokenizer == mock_tokenizer
        assert model.tokenizer == mock_tokenizer
        assert mock_tokenizer.pad_token == "<eos>"
        assert mock_tokenizer.pad_token_id == 1

    @patch("src.models.lora_model.AutoTokenizer")
    def test_load_tokenizer_with_pad_token(self, mock_tokenizer_class, sample_config):
        """Test loading tokenizer that already has pad token."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        model = LoRAModel(sample_config)
        loaded_tokenizer = model.load_tokenizer()

        assert loaded_tokenizer == mock_tokenizer
        assert model.tokenizer == mock_tokenizer
        # Should not change existing pad token
        assert mock_tokenizer.pad_token == "<pad>"
        assert mock_tokenizer.pad_token_id == 0

    @patch("src.models.lora_model.get_peft_model")
    @patch("src.models.lora_model.LoraConfig")
    def test_setup_lora(
        self, mock_lora_config, mock_get_peft, sample_config, mock_model
    ):
        """Test setting up LoRA configuration."""
        mock_peft_model = Mock()
        mock_peft_model.train.return_value = None
        mock_peft_model.named_parameters.return_value = [
            ("lora_A", torch.randn(10, 10)),
            ("base_model", torch.randn(20, 20)),
        ]
        mock_peft_model.print_trainable_parameters.return_value = None
        mock_get_peft.return_value = mock_peft_model

        model = LoRAModel(sample_config)
        model.model = mock_model

        peft_model = model.setup_lora()

        assert peft_model == mock_peft_model
        assert model.peft_model == mock_peft_model
        mock_lora_config.assert_called_once()
        mock_get_peft.assert_called_once()

    def test_setup_lora_no_model(self, sample_config):
        """Test setting up LoRA without loaded model."""
        model = LoRAModel(sample_config)

        with pytest.raises(
            ValueError, match="Model must be loaded before setting up LoRA"
        ):
            model.setup_lora()

    def test_get_model_with_peft(self, sample_config, mock_peft_model):
        """Test getting model when PEFT model is available."""
        model = LoRAModel(sample_config)
        model.peft_model = mock_peft_model

        result = model.get_model()
        assert result == mock_peft_model

    def test_get_model_with_base_model(self, sample_config, mock_model):
        """Test getting model when only base model is available."""
        model = LoRAModel(sample_config)
        model.model = mock_model

        result = model.get_model()
        assert result == mock_model

    def test_get_model_no_model(self, sample_config):
        """Test getting model when no model is loaded."""
        model = LoRAModel(sample_config)

        with pytest.raises(ValueError, match="Model not loaded"):
            model.get_model()

    def test_get_tokenizer(self, sample_config, mock_tokenizer):
        """Test getting tokenizer."""
        model = LoRAModel(sample_config)
        model.tokenizer = mock_tokenizer

        result = model.get_tokenizer()
        assert result == mock_tokenizer

    def test_get_tokenizer_no_tokenizer(self, sample_config):
        """Test getting tokenizer when not loaded."""
        model = LoRAModel(sample_config)

        with pytest.raises(ValueError, match="Tokenizer not loaded"):
            model.get_tokenizer()

    def test_save_model_peft(self, sample_config, mock_peft_model, temp_dir):
        """Test saving PEFT model."""
        model = LoRAModel(sample_config)
        model.peft_model = mock_peft_model
        model.tokenizer = Mock()

        output_dir = temp_dir / "model_output"
        model.save_model(output_dir, save_tokenizer=True)

        mock_peft_model.save_pretrained.assert_called_once_with(output_dir)
        model.tokenizer.save_pretrained.assert_called_once_with(output_dir)

    def test_save_model_base(self, sample_config, mock_model, temp_dir):
        """Test saving base model."""
        model = LoRAModel(sample_config)
        model.model = mock_model
        model.tokenizer = Mock()

        output_dir = temp_dir / "model_output"
        model.save_model(output_dir, save_tokenizer=False)

        mock_model.save_pretrained.assert_called_once_with(output_dir)
        model.tokenizer.save_pretrained.assert_not_called()

    @patch("src.models.lora_model.PeftModel")
    def test_load_from_checkpoint(
        self, mock_peft_model_class, sample_config, mock_model, temp_dir
    ):
        """Test loading model from checkpoint."""
        checkpoint_path = temp_dir / "checkpoint"
        mock_peft_model = Mock()
        mock_peft_model_class.from_pretrained.return_value = mock_peft_model

        model = LoRAModel(sample_config)
        model.model = mock_model

        with patch.object(model, "setup_lora") as mock_setup_lora:
            model.load_from_checkpoint(checkpoint_path)

        mock_peft_model_class.from_pretrained.assert_called_once()
        assert model.peft_model == mock_peft_model

    def test_merge_and_unload(self, sample_config, mock_peft_model, mock_model):
        """Test merging and unloading LoRA."""
        mock_peft_model.merge_and_unload.return_value = mock_model

        model = LoRAModel(sample_config)
        model.peft_model = mock_peft_model

        result = model.merge_and_unload()

        assert result == mock_model
        mock_peft_model.merge_and_unload.assert_called_once()

    def test_merge_and_unload_no_peft(self, sample_config):
        """Test merging and unloading when no PEFT model."""
        model = LoRAModel(sample_config)

        with pytest.raises(ValueError, match="No LoRA model to merge"):
            model.merge_and_unload()

    def test_get_trainable_parameters(self, sample_config, mock_peft_model):
        """Test getting trainable parameters info."""
        # Mock parameters
        trainable_param = torch.randn(10, 10)
        trainable_param.requires_grad = True
        non_trainable_param = torch.randn(20, 20)
        non_trainable_param.requires_grad = False

        mock_peft_model.named_parameters.return_value = [
            ("lora_A", trainable_param),
            ("base_model", non_trainable_param),
        ]

        model = LoRAModel(sample_config)
        model.peft_model = mock_peft_model

        info = model.get_trainable_parameters()

        assert "total" in info
        assert "trainable" in info
        assert "percentage" in info
        assert info["total"] == 10 * 10 + 20 * 20
        assert info["trainable"] == 10 * 10
        assert info["percentage"] == (10 * 10) / (10 * 10 + 20 * 20) * 100

    def test_get_trainable_parameters_no_peft(self, sample_config):
        """Test getting trainable parameters when no PEFT model."""
        model = LoRAModel(sample_config)

        info = model.get_trainable_parameters()

        assert info == {"total": 0, "trainable": 0, "percentage": 0.0}

    def test_generate(self, sample_config, mock_peft_model, mock_tokenizer):
        """Test text generation."""
        input_ids = torch.randint(0, 1000, (1, 10))
        output_ids = torch.randint(0, 1000, (1, 20))
        mock_peft_model.generate.return_value = output_ids

        model = LoRAModel(sample_config)
        model.peft_model = mock_peft_model
        model.tokenizer = mock_tokenizer
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        result = model.generate(input_ids, max_new_tokens=10)

        assert torch.equal(result, output_ids)
        mock_peft_model.generate.assert_called_once()

    def test_generate_with_tokenizer_ids(self, sample_config, mock_peft_model):
        """Test text generation with explicit tokenizer IDs."""
        input_ids = torch.randint(0, 1000, (1, 10))
        output_ids = torch.randint(0, 1000, (1, 20))
        mock_peft_model.generate.return_value = output_ids

        model = LoRAModel(sample_config)
        model.peft_model = mock_peft_model

        result = model.generate(
            input_ids, max_new_tokens=10, pad_token_id=0, eos_token_id=1
        )

        assert torch.equal(result, output_ids)
        mock_peft_model.generate.assert_called_once()


class TestCreateLoRAModel:
    """Test cases for create_lora_model factory function."""

    @patch("src.models.lora_model.LoRAModel")
    def test_create_lora_model(self, mock_model_class, sample_config):
        """Test creating LoRA model using factory function."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        result = create_lora_model(sample_config)

        assert result == mock_model
        mock_model.load_model.assert_called_once()
        mock_model.load_tokenizer.assert_called_once()
        mock_model.setup_lora.assert_called_once()


class TestLoRAModelEdgeCases:
    """Test edge cases and error handling."""

    def test_model_with_trust_remote_code(self, sample_config):
        """Test loading model with trust_remote_code=True."""
        sample_config.trust_remote_code = True

        with patch("src.models.lora_model.AutoModelForCausalLM") as mock_model_class:
            mock_model = Mock()
            mock_model_class.from_pretrained.return_value = mock_model

            model = LoRAModel(sample_config)
            model.load_model()

            # Check that trust_remote_code was passed
            call_args = mock_model_class.from_pretrained.call_args
            assert call_args[1]["trust_remote_code"] is True

    def test_model_with_custom_torch_dtype(self, sample_config):
        """Test loading model with custom torch dtype."""
        sample_config.torch_dtype = "float16"

        with patch("src.models.lora_model.AutoModelForCausalLM") as mock_model_class:
            mock_model = Mock()
            mock_model_class.from_pretrained.return_value = mock_model

            model = LoRAModel(sample_config)
            model.load_model()

            # Check that torch_dtype was passed
            call_args = mock_model_class.from_pretrained.call_args
            assert call_args[1]["torch_dtype"] == torch.float16

    def test_model_with_auto_torch_dtype(self, sample_config):
        """Test loading model with auto torch dtype."""
        sample_config.torch_dtype = "auto"

        with patch("src.models.lora_model.AutoModelForCausalLM") as mock_model_class:
            mock_model = Mock()
            mock_model_class.from_pretrained.return_value = mock_model

            model = LoRAModel(sample_config)
            model.load_model()

            # Check that torch_dtype was passed as "auto"
            call_args = mock_model_class.from_pretrained.call_args
            assert call_args[1]["torch_dtype"] == "auto"

    def test_model_with_custom_device_map(self, sample_config):
        """Test loading model with custom device map."""
        sample_config.device_map = "cuda:0"

        with patch("src.models.lora_model.AutoModelForCausalLM") as mock_model_class:
            mock_model = Mock()
            mock_model_class.from_pretrained.return_value = mock_model

            model = LoRAModel(sample_config)
            model.load_model()

            # Check that device_map was passed
            call_args = mock_model_class.from_pretrained.call_args
            assert call_args[1]["device_map"] == "cuda:0"

    def test_tokenizer_with_trust_remote_code(self, sample_config):
        """Test loading tokenizer with trust_remote_code=True."""
        sample_config.trust_remote_code = True

        with patch("src.models.lora_model.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "<eos>"
            mock_tokenizer.eos_token_id = 1
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            model = LoRAModel(sample_config)
            model.load_tokenizer()

            # Check that trust_remote_code was passed
            call_args = mock_tokenizer_class.from_pretrained.call_args
            assert call_args[1]["trust_remote_code"] is True

    def test_save_model_creates_directory(
        self, sample_config, mock_peft_model, temp_dir
    ):
        """Test that save_model creates output directory."""
        output_dir = temp_dir / "new_directory" / "model"

        model = LoRAModel(sample_config)
        model.peft_model = mock_peft_model

        model.save_model(output_dir)

        assert output_dir.exists()
        mock_peft_model.save_pretrained.assert_called_once_with(output_dir)

    def test_generate_with_kwargs(self, sample_config, mock_peft_model):
        """Test text generation with additional kwargs."""
        input_ids = torch.randint(0, 1000, (1, 10))
        output_ids = torch.randint(0, 1000, (1, 20))
        mock_peft_model.generate.return_value = output_ids

        model = LoRAModel(sample_config)
        model.peft_model = mock_peft_model

        result = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            custom_param="test",
        )

        assert torch.equal(result, output_ids)

        # Check that all parameters were passed
        call_args = mock_peft_model.generate.call_args
        assert call_args[1]["temperature"] == 0.8
        assert call_args[1]["top_p"] == 0.9
        assert call_args[1]["top_k"] == 50
        assert call_args[1]["do_sample"] is True
        assert call_args[1]["custom_param"] == "test"
