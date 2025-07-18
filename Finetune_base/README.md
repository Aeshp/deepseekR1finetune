# FineTune_base Scripts

This folder contains Python scripts for fine-tuning and inference using the DeepSeek-R1-Distill-Llama-8B model with Unsloth and related libraries.

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
   The scripts expect a Hugging Face dataset (e.g., `taskydata/baize_chatbot`).

## Scripts

### 1. `unsloth_deepseek_basetune.py`
- This script fine-tunes the DeepSeek-R1-Distill-Llama-8B model using Unsloth and LoRA.
- It loads the dataset, formats it, sets up TensorBoard logging, and starts training.
- **Note:** Inference code is separated for clarity.

#### Usage:
```bash
python unsloth_deepseek_basetune.py
```

### 2. `inferance.py`
- This script runs inference using your trained model and tokenizer.
- Make sure to load the same model and tokenizer as used in training.

#### Usage:
```bash
python inferance.py
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
- Adjust batch size, gradient accumulation, and sequence length in the script according to your GPU/VRAM.
- For large datasets, ensure you have enough disk and memory resources.
- For more details on Unsloth and DeepSeek, see their official documentation.
