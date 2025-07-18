# Hugging Face Model Push Scripts

This folder contains scripts for pushing models and weights to the Hugging Face Hub. Please read below to understand the requirements and workflow for each script.

## Requirements
Add these to your `requirements.txt` before running any script:
```
huggingface_hub
peft
transformers
```
If you use Jupyter notebooks, also add:
```
ipython
```

Install with:
```
pip install -r requirements.txt
```

## Workflow

### 1. Fine-tune or Load Your Model
Before using these scripts, you must have a trained or loaded model and tokenizer objects in your Python environment:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("your_model_path")
tokenizer = AutoTokenizer.from_pretrained("your_model_path")
```
Or, after fine-tuning, use your trained model and tokenizer.

---

### 2. `push_only_weights.py`
This script pushes only the LoRA adapter weights and tokenizer to the Hugging Face Hub. Use this if you want to share just the fine-tuned adapter weights (not the full merged model).
- **Resource usage:** Low (only uploads adapter weights)
- **Required libraries:** `huggingface_hub`, `transformers`
- **Usage:**
    1. Log in to Hugging Face Hub.
    2. Push model and tokenizer using `model.push_to_hub()` and `tokenizer.push_to_hub()`.

---

### 3. `merge_weights_and_push.py`
This script merges the LoRA adapter weights into the base model, then pushes the full merged model and tokenizer to the Hugging Face Hub. Use this if you want to share the complete model (base + adapters merged).
- **Resource usage:** High (merging and uploading full model requires more RAM/VRAM and disk space)
- **Required libraries:** `peft`, `huggingface_hub`, `transformers`
- **Usage:**
    1. Log in to Hugging Face Hub.
    2. Merge LoRA adapters into the base model using `model.merge_and_unload()`.
    3. Save the merged model and tokenizer locally.
    4. Push the merged model folder to the Hub using `HfApi().upload_folder()`.

---

## Notes
- Make sure your `model` and `tokenizer` objects are loaded before running these scripts.
- For large models, merging and uploading may require significant resources (RAM, VRAM, disk space).
- Replace `Yourusername/modelrepo` with your actual Hugging Face username and repository name.
- You must be logged in to Hugging Face Hub using `notebook_login()` or `login()`.

---

For more details, see the official Hugging Face documentation: https://huggingface.co/docs/hub/index
