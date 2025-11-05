"""
Test script to verify all imports and CUDA availability.
This script tests imports from both training and generation scripts.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("Testing Imports and CUDA Availability")
print("=" * 80)
print()

# Track failures
failures = []
successes = []


def test_import(name, module_path=None):
    """Test importing a module and return success status."""
    try:
        if module_path:
            __import__(module_path)
            print(f"✓ {name}")
            successes.append(name)
            return True
        else:
            __import__(name)
            print(f"✓ {name}")
            successes.append(name)
            return True
    except ImportError as e:
        print(f"✗ {name}: {e}")
        failures.append(f"{name}: {e}")
        return False
    except Exception as e:
        print(f"✗ {name}: {type(e).__name__}: {e}")
        failures.append(f"{name}: {type(e).__name__}: {e}")
        return False


print("Testing standard library imports...")
print("-" * 80)
test_import("sys")
test_import("pathlib")
test_import("logging")
print()

print("Testing third-party library imports...")
print("-" * 80)
test_import("hydra")
test_import("omegaconf")
test_import("torch")
test_import("transformers")
test_import("trl")
print()

print("Testing imports from 06_train_model.py...")
print("-" * 80)
test_import("hydra", "hydra")
test_import("omegaconf.DictConfig", "omegaconf")
test_import("omegaconf.OmegaConf", "omegaconf")
test_import("torch", "torch")
test_import("transformers.TrainingArguments", "transformers")
test_import("transformers.EarlyStoppingCallback", "transformers")
test_import("trl.SFTTrainer", "trl")
print()

print("Testing project-specific imports from 06_train_model.py...")
print("-" * 80)
test_import("src.data.collator", "src.data.collator")
test_import("src.training.trainer", "src.training.trainer")
test_import("src.utils.model_utils", "src.utils.model_utils")
test_import("src.training.callbacks", "src.training.callbacks")
test_import("src.models.initialization", "src.models.initialization")
test_import("src.data.loading", "src.data.loading")
print()

print("Testing imports from 08_generate_samples.py...")
print("-" * 80)
test_import("src.models.modular_multimodal_model", "src.models.modular_multimodal_model")
test_import("src.models.checkpoint_loading", "src.models.checkpoint_loading")
test_import("src.evaluation.generation", "src.evaluation.generation")
print()

print("Testing specific class imports...")
print("-" * 80)
try:
    from src.data.collator import MultimodalDataCollator
    print("✓ MultimodalDataCollator")
    successes.append("MultimodalDataCollator")
except Exception as e:
    print(f"✗ MultimodalDataCollator: {e}")
    failures.append(f"MultimodalDataCollator: {e}")

try:
    from src.training.trainer import ExperimentTrackingCallback
    print("✓ ExperimentTrackingCallback")
    successes.append("ExperimentTrackingCallback")
except Exception as e:
    print(f"✗ ExperimentTrackingCallback: {e}")
    failures.append(f"ExperimentTrackingCallback: {e}")

try:
    from src.utils.model_utils import initialize_experiment_tracker, load_dataset, find_latest_checkpoint
    print("✓ initialize_experiment_tracker, load_dataset, find_latest_checkpoint")
    successes.append("model_utils functions")
except Exception as e:
    print(f"✗ model_utils functions: {e}")
    failures.append(f"model_utils functions: {e}")

try:
    from src.training.callbacks import ProjectionDiagnosticCallback
    print("✓ ProjectionDiagnosticCallback")
    successes.append("ProjectionDiagnosticCallback")
except Exception as e:
    print(f"✗ ProjectionDiagnosticCallback: {e}")
    failures.append(f"ProjectionDiagnosticCallback: {e}")

try:
    from src.models.initialization import initialize_model_and_tokenizer
    print("✓ initialize_model_and_tokenizer")
    successes.append("initialize_model_and_tokenizer")
except Exception as e:
    print(f"✗ initialize_model_and_tokenizer: {e}")
    failures.append(f"initialize_model_and_tokenizer: {e}")

try:
    from src.data.loading import load_datasets
    print("✓ load_datasets")
    successes.append("load_datasets")
except Exception as e:
    print(f"✗ load_datasets: {e}")
    failures.append(f"load_datasets: {e}")

try:
    from src.models.modular_multimodal_model import ModularMultimodalModel
    print("✓ ModularMultimodalModel")
    successes.append("ModularMultimodalModel")
except Exception as e:
    print(f"✗ ModularMultimodalModel: {e}")
    failures.append(f"ModularMultimodalModel: {e}")

try:
    from src.models.checkpoint_loading import load_trained_model
    print("✓ load_trained_model")
    successes.append("load_trained_model")
except Exception as e:
    print(f"✗ load_trained_model: {e}")
    failures.append(f"load_trained_model: {e}")

try:
    from src.evaluation.generation import generate_and_compare
    print("✓ generate_and_compare")
    successes.append("generate_and_compare")
except Exception as e:
    print(f"✗ generate_and_compare: {e}")
    failures.append(f"generate_and_compare: {e}")

print()
print("=" * 80)
print("Testing CUDA Availability")
print("=" * 80)
print()

try:
    import torch
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        successes.append("CUDA")
    else:
        print("⚠ CUDA is not available")
        failures.append("CUDA not available")
        
except Exception as e:
    print(f"✗ Error checking CUDA: {e}")
    failures.append(f"CUDA check: {e}")

print()
print("=" * 80)
print("Summary")
print("=" * 80)
print(f"Total successful imports: {len(successes)}")
print(f"Total failed imports: {len(failures)}")
print()

if failures:
    print("FAILURES:")
    for failure in failures:
        print(f"  - {failure}")
    print()
    print("❌ Some imports or CUDA checks failed!")
    sys.exit(1)
else:
    print("✅ All imports and CUDA checks passed!")
    sys.exit(0)

