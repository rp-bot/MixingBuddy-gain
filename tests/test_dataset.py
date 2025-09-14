"""
Tests for dataset functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from src.data.dataset import (
    AutomaticMixingDataset,
    DataProcessor,
    create_sample_data,
    load_huggingface_dataset,
)


class TestAutomaticMixingDataset:
    """Test cases for AutomaticMixingDataset class."""

    def test_init(self, sample_data, mock_tokenizer):
        """Test dataset initialization."""
        dataset = AutomaticMixingDataset(
            data=sample_data,
            tokenizer=mock_tokenizer,
            max_length=256,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )

        assert len(dataset) == len(sample_data)
        assert dataset.max_length == 256
        assert dataset.padding == "max_length"
        assert dataset.truncation is True
        assert dataset.add_special_tokens is True

    def test_getitem(self, sample_data, mock_tokenizer):
        """Test getting dataset items."""
        dataset = AutomaticMixingDataset(
            data=sample_data,
            tokenizer=mock_tokenizer,
            max_length=128,
        )

        # Test first item
        item = dataset[0]

        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

        # Check tensor shapes
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

        # Labels should be same as input_ids for causal LM
        assert torch.equal(item["input_ids"], item["labels"])

    def test_format_text_instruction_response(self, mock_tokenizer):
        """Test text formatting for instruction-response format."""
        dataset = AutomaticMixingDataset(
            data=[{"instruction": "Test instruction", "response": "Test response"}],
            tokenizer=mock_tokenizer,
        )

        formatted = dataset._format_text(
            {"instruction": "Test instruction", "response": "Test response"}
        )
        expected = "### Instruction:\nTest instruction\n\n### Response:\nTest response"
        assert formatted == expected

    def test_format_text_input_output(self, mock_tokenizer):
        """Test text formatting for input-output format."""
        dataset = AutomaticMixingDataset(
            data=[{"input": "Test input", "output": "Test output"}],
            tokenizer=mock_tokenizer,
        )

        formatted = dataset._format_text(
            {"input": "Test input", "output": "Test output"}
        )
        expected = "Input: Test input\nOutput: Test output"
        assert formatted == expected

    def test_format_text_plain(self, mock_tokenizer):
        """Test text formatting for plain text format."""
        dataset = AutomaticMixingDataset(
            data=[{"text": "Plain text content"}],
            tokenizer=mock_tokenizer,
        )

        formatted = dataset._format_text({"text": "Plain text content"})
        assert formatted == "Plain text content"

    def test_format_text_fallback(self, mock_tokenizer):
        """Test text formatting fallback."""
        dataset = AutomaticMixingDataset(
            data=[{"key1": "value1", "key2": "value2"}],
            tokenizer=mock_tokenizer,
        )

        formatted = dataset._format_text({"key1": "value1", "key2": "value2"})
        expected = "key1: value1 key2: value2"
        assert formatted == expected

    def test_from_jsonl(self, temp_data_file, mock_tokenizer):
        """Test loading dataset from JSONL file."""
        dataset = AutomaticMixingDataset.from_jsonl(
            temp_data_file,
            tokenizer=mock_tokenizer,
            max_length=256,
        )

        assert isinstance(dataset, AutomaticMixingDataset)
        assert len(dataset) > 0

    def test_from_json(self, sample_data, mock_tokenizer, temp_dir):
        """Test loading dataset from JSON file."""
        # Create JSON file
        json_file = temp_dir / "test.json"
        with open(json_file, "w") as f:
            json.dump(sample_data, f)

        dataset = AutomaticMixingDataset.from_json(
            json_file,
            tokenizer=mock_tokenizer,
            max_length=256,
        )

        assert isinstance(dataset, AutomaticMixingDataset)
        assert len(dataset) == len(sample_data)

    def test_from_json_dict_format(self, mock_tokenizer, temp_dir):
        """Test loading dataset from JSON file with dict format."""
        # Create JSON file with dict format
        json_data = {"data": [{"text": "sample 1"}, {"text": "sample 2"}]}
        json_file = temp_dir / "test.json"
        with open(json_file, "w") as f:
            json.dump(json_data, f)

        dataset = AutomaticMixingDataset.from_json(
            json_file,
            tokenizer=mock_tokenizer,
            max_length=256,
        )

        assert isinstance(dataset, AutomaticMixingDataset)
        assert len(dataset) == 2


class TestDataProcessor:
    """Test cases for DataProcessor class."""

    def test_init(self, sample_config):
        """Test DataProcessor initialization."""
        processor = DataProcessor(sample_config)
        assert processor.config == sample_config

    @patch("src.data.dataset.AutoTokenizer")
    def test_load_dataset(self, mock_tokenizer_class, sample_config, temp_dir):
        """Test loading dataset."""
        # Setup
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Create test data files
        data_dir = temp_dir / "processed"
        data_dir.mkdir()

        train_file = data_dir / "train.jsonl"
        with open(train_file, "w") as f:
            f.write('{"instruction": "test", "response": "test"}\n')

        # Update config
        sample_config.data.dataset.path = str(data_dir)

        processor = DataProcessor(sample_config)

        with patch.object(processor, "config", sample_config):
            dataset = processor.load_dataset("train")

        assert isinstance(dataset, AutomaticMixingDataset)

    def test_create_dataloader(self, sample_config, sample_data, mock_tokenizer):
        """Test creating DataLoader."""
        dataset = AutomaticMixingDataset(data=sample_data, tokenizer=mock_tokenizer)
        processor = DataProcessor(sample_config)

        dataloader = processor.create_dataloader(dataset, "train")

        assert hasattr(dataloader, "batch_size")
        assert hasattr(dataloader, "dataset")

    def test_get_dataset_stats(self, sample_config, sample_data, mock_tokenizer):
        """Test getting dataset statistics."""
        dataset = AutomaticMixingDataset(data=sample_data, tokenizer=mock_tokenizer)
        processor = DataProcessor(sample_config)

        stats = processor.get_dataset_stats(dataset)

        assert "num_samples" in stats
        assert "avg_length" in stats
        assert "max_length" in stats
        assert "min_length" in stats
        assert "total_tokens" in stats
        assert stats["num_samples"] == len(sample_data)


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_create_sample_data(self, temp_dir):
        """Test creating sample data."""
        output_dir = temp_dir / "sample_data"
        create_sample_data(output_dir, num_samples=10)

        # Check that files were created
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "validation.jsonl").exists()
        assert (output_dir / "test.jsonl").exists()

        # Check content
        with open(output_dir / "train.jsonl") as f:
            train_data = [json.loads(line) for line in f]

        assert len(train_data) > 0
        assert "instruction" in train_data[0]
        assert "response" in train_data[0]

    @patch("src.data.dataset.load_dataset")
    @patch("src.data.dataset.AutoTokenizer")
    def test_load_huggingface_dataset(
        self, mock_tokenizer_class, mock_load_dataset, mock_tokenizer
    ):
        """Test loading Hugging Face dataset."""
        # Setup mocks
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_hf_dataset = Mock()
        mock_hf_dataset.__iter__ = Mock(
            return_value=iter([{"text": "sample 1"}, {"text": "sample 2"}])
        )
        mock_load_dataset.return_value = mock_hf_dataset

        dataset = load_huggingface_dataset("test-dataset", tokenizer=mock_tokenizer)

        assert isinstance(dataset, AutomaticMixingDataset)
        mock_load_dataset.assert_called_once_with("test-dataset", split="train")


class TestDatasetEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self, mock_tokenizer):
        """Test handling of empty dataset."""
        dataset = AutomaticMixingDataset(data=[], tokenizer=mock_tokenizer)
        assert len(dataset) == 0

    def test_invalid_json_file(self, mock_tokenizer, temp_dir):
        """Test handling of invalid JSON file."""
        invalid_file = temp_dir / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("invalid json content")

        with pytest.raises(json.JSONDecodeError):
            AutomaticMixingDataset.from_json(invalid_file, tokenizer=mock_tokenizer)

    def test_missing_data_file(self, mock_tokenizer):
        """Test handling of missing data file."""
        with pytest.raises(FileNotFoundError):
            AutomaticMixingDataset.from_jsonl(
                "nonexistent.jsonl", tokenizer=mock_tokenizer
            )

    def test_unsupported_data_format(self, mock_tokenizer, temp_dir):
        """Test handling of unsupported data format in JSON."""
        json_file = temp_dir / "test.json"
        with open(json_file, "w") as f:
            json.dump("invalid format", f)  # String instead of list/dict

        with pytest.raises(ValueError, match="Unexpected data format"):
            AutomaticMixingDataset.from_json(json_file, tokenizer=mock_tokenizer)
