# finetune_checkpoint

This folder contains scripts and resources for further fine-tuning Aeshp/deepseekR1tunedchat a DeepSeek/Unsloth-based model from a checkpoint, as well as running inference and pushing weights to Hugging Face.

## Overview

- **This example demonstrates fine-tuning on a small dataset for illustration purposes.**
- For best results, use a larger and more diverse dataset for both training and evaluation.

## Contents

- `requirements.txt` — All required Python packages for this workflow.
- `deepseekR1tunedchat.py` — Script for further fine-tuning from a checkpoint using Unsloth, LoRA, and Hugging Face tools.
- `inferance.py` — Script for running inference with your fine-tuned model.
- `push_weights.py` — Script to push your model weights to the Hugging Face Hub.

## Setup

1. **Install requirements:**
   Make sure you have Python 3.8+ and run:
   ```bash
   pip install -r requirements.txt
   ```
   This will install all necessary packages, including:
   - transformers
   - peft
   - unsloth
   - trl
   - huggingface_hub
   - bitsandbytes
   - accelerate
   - datasets
   - torch
   - tensorboard
   - matplotlib
   - ipython
   - jupyter

2. **Prepare your dataset:**
   - The scripts expect your training and evaluation data in JSONL format (see `train.jsonl` and `test.jsonl` examples).
   - For best results, use as much high-quality data as possible. This example uses a small dataset for demonstration only.

## Scripts

### 1. `deepseekR1tunedchat.py`
- Further fine-tunes a model from a checkpoint using Unsloth and LoRA.
- Loads your dataset, sets up logging, and starts training.
- Adjust hyperparameters in the script as needed.

#### Usage:
```bash
python deepseekR1tunedchat.py
```

### 2. `inferance.py`
- Runs inference using your fine-tuned model and tokenizer.
- Make sure to load the same model and tokenizer as used in training.

#### Usage:
```bash
python inferance.py
```

### 3. `push_weights.py`
- Pushes your model weights to the Hugging Face Hub.
- Make sure you are logged in to your Hugging Face account.

#### Usage:
```bash
python push_weights.py
```

## Using TensorBoard

During training, logs are saved to the directory specified in the script (e.g., `outputs/runs/`).
To view training progress:

1. Open a terminal in VS Code (or your environment).
2. Run:
   ```bash
   tensorboard --logdir outputs/runs/
   ```
3. Open the link shown in the terminal (usually http://localhost:6006) in your browser.

You can also install the TensorBoard extension in VS Code to view logs directly inside the editor.

## Notes
- This example uses a small dataset for demonstration. For real applications, use a larger dataset for both training and evaluation.
- Adjust batch size, gradient accumulation, and sequence length in the script according to your GPU/VRAM.
- Do not include your `venv` folder or other large local files when pushing to Hugging Face or version control.
- For more details on Unsloth, DeepSeek, and Hugging Face workflows, see their official documentation.
