# LLM Fine-tuning with LoRA for Automatic Mixing Research

A comprehensive, production-ready framework for fine-tuning Large Language Models (LLMs) using LoRA (Low-Rank Adaptation) for automatic mixing applications in music production. This project emphasizes robust MLOps workflows, including experiment tracking and code modularization.

## 🚀 Features

- **LoRA Fine-tuning**: Efficient fine-tuning using Low-Rank Adaptation
- **Modular Architecture**: Clean, extensible codebase with separation of concerns
- **Experiment Tracking**: Integration with Weights & Biases and MLflow
- **Configuration Management**: Hydra-based configuration system
- **Comprehensive Evaluation**: Domain-specific metrics for automatic mixing
- **Production Ready**: Docker support and deployment scripts

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   ├── models/                   # Model definitions and LoRA implementation
│   ├── training/                 # Training pipeline and utilities
│   ├── evaluation/               # Evaluation metrics and testing
│   └── utils/                    # Utility functions and helpers
├── configs/                      # Configuration files
│   ├── model/                    # Model configurations
│   ├── data/                     # Data configurations
│   ├── training/                 # Training configurations
│   └── experiment_tracking/      # Experiment tracking configs
├── scripts/                      # Executable scripts
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── inference.py              # Inference script
│   └── prepare_data.py           # Data preparation script
├── tests/                        # Test suite
├── pyproject.toml                # Project configuration
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
└── Makefile                      # Build automation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Quick Start

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd automatic-mixing/milestone_0
   ```

2. **Set up virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   # Production dependencies
   pip install -r requirements.txt

   # Development dependencies (optional)
   pip install -r requirements-dev.txt
   ```

4. **Set up development environment**
   ```bash
   make install-dev
   ```

### Docker Installation

```bash
# Build Docker image
docker build -t llm-lora-finetuning .

# Run with GPU support
docker run --gpus all -it llm-lora-finetuning
```

## 🚀 Quick Start

### 1. Prepare Data

```bash
# Create sample data (for testing)
make prepare-data

# Or use your own data
# Place your data in data/raw/ directory
```

### 2. Train a Model

```bash
# Train with default configuration
make train

# Train with specific model
python scripts/train.py model=llama2_7b

# Train with custom parameters
python scripts/train.py \
  model=llama2_13b \
  training.training_args.num_train_epochs=5 \
  training.training_args.learning_rate=1e-4
```

### 3. Evaluate Model

```bash
# Evaluate trained model
make evaluate

# Evaluate with specific checkpoint
python scripts/evaluate.py checkpoint_path=outputs/checkpoints/final_model
```

### 4. Run Inference

```bash
# Run inference with example prompts
make infer

# Interactive inference
python scripts/inference.py --interactive
```

## 📊 Data Format

The framework expects data in JSONL format with the following structure:

```json
{"instruction": "Analyze the mixing parameters for this track", "response": "Track requires EQ adjustments at 2kHz, compression with ratio 3:1, and reverb with 1.2s decay time."}
{"instruction": "What EQ adjustments should I make for a vocal track?", "response": "For vocals, apply high-pass filter at 80Hz, gentle boost at 2-5kHz for presence, and slight cut at 300-500Hz to reduce muddiness."}
```

### Data Directory Structure

```
data/
├── raw/                          # Raw data files
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
└── processed/                    # Processed data (auto-generated)
    ├── train.jsonl
    ├── validation.jsonl
    └── test.jsonl
```

## ⚙️ Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/`: Model-specific configurations
- `configs/training/`: Training configurations
- `configs/data/`: Data processing configurations

### Example Configuration Override

```bash
python scripts/train.py \
  model=llama2_7b \
  training.training_args.learning_rate=2e-4 \
  training.training_args.per_device_train_batch_size=8 \
  data.processing.max_length=1024
```

## 🔬 Experiment Tracking

### Weights & Biases

```bash
# Train with WandB tracking
python scripts/train.py experiment_tracking=wandb

# Set WandB project name
python scripts/train.py \
  experiment_tracking=wandb \
  experiment_tracking.project=my-automatic-mixing-project
```

### MLflow

```bash
# Train with MLflow tracking
python scripts/train.py experiment_tracking=mlflow
```

## 📊 Data Management

### Data Directory Structure

```
data/
├── raw/                          # Raw data files
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
└── processed/                    # Processed data (auto-generated)
    ├── train.jsonl
    ├── validation.jsonl
    └── test.jsonl
```

### Data Operations

```bash
# Prepare data from raw files
python scripts/prepare_data.py

# Check data statistics
python -c "from src.data.dataset import DataProcessor; dp = DataProcessor(config); print(dp.get_dataset_stats(dp.load_dataset('train')))"
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔍 Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Run all quality checks
make ci-test
```

## 🔄 Development Workflow

The project supports a streamlined development workflow:

### Local Development

- Code quality checks with linting and formatting
- Automated testing with pytest
- Type checking with mypy
- Docker-based development environment

### Usage

```bash
# Run code quality checks
make lint
make format
make type-check

# Run tests
make test

# Full development workflow
make ci-test
```

## 📊 Evaluation Metrics

The framework includes domain-specific metrics for automatic mixing:

- **Text Metrics**: BLEU, ROUGE, BERT Score
- **Parameter Accuracy**: EQ, compression, reverb parameter extraction
- **Technical Term Accuracy**: Mixing terminology recognition
- **Parameter Value Accuracy**: Numerical parameter prediction accuracy

## 🐳 Docker Support

### Build and Run

```bash
# Build image
docker build -t llm-lora-finetuning .

# Run with GPU support
docker run --gpus all -it llm-lora-finetuning

# Run training
docker run --gpus all -v $(pwd)/data:/app/data llm-lora-finetuning python scripts/train.py
```

## 📚 API Reference

### Core Classes

#### `LoRAModel`

Main model wrapper for LoRA fine-tuning.

```python
from src.models.lora_model import LoRAModel

model = LoRAModel(config)
model.load_model()
model.load_tokenizer()
model.setup_lora()
```

#### `LoRATrainer`

Training pipeline with experiment tracking.

```python
from src.training.trainer import LoRATrainer

trainer = LoRATrainer(model, config, experiment_tracker)
trainer.train(train_dataloader, val_dataloader)
```

#### `ExperimentTracker`

Unified experiment tracking interface.

```python
from src.utils.experiment_tracking import ExperimentTracker

tracker = ExperimentTracker(config, backend="wandb")
tracker.log_metrics({"loss": 0.5})
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone <your-fork-url>
cd automatic-mixing/milestone_0

# Set up development environment
make install-dev

# Run code quality checks
make format
make lint
make type-check
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [Hydra Configuration Framework](https://hydra.cc/)
- [Weights & Biases](https://wandb.ai/)
- [MLflow](https://mlflow.org/)

## 📞 Support

For questions and support:

- Create an issue on GitHub
- Check the documentation
- Review the example configurations

## 🔮 Roadmap

- [ ] Support for more model architectures
- [ ] Advanced data augmentation techniques
- [ ] Model compression and quantization
- [ ] Real-time inference API
- [ ] Multi-GPU training support
- [ ] Model serving with FastAPI
- [ ] Integration with music production software

---

**Happy Fine-tuning! 🎵🤖**
