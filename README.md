# LLM Fine-tuning with LoRA for Automatic Mixing Research

## Introduction

This repository provides a minimal, production-ready setup to fine-tune Large Language Models (LLMs) with LoRA (Low-Rank Adaptation) for automatic mixing research in music production. It focuses on clarity and practicality so you can get set up quickly and start experimenting.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU and drivers (optional, recommended)

### Steps

1. Clone the repository

   ```bash
   git clone <repository-url>
   cd MixingBuddy-gain
   ```

2. Create and activate a virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

You’re ready to go. See scripts in `scripts/` and configs in `configs/` to begin training and experimentation. You also need the dataset in `data/` to train the model. and the checkpoints in `checkpoints/` to evaluate or inference the model.
