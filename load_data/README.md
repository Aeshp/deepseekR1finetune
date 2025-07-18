# Load Data Scripts

This folder contains example scripts for loading datasets either from the Hugging Face Hub or from local/Colab files. Use these scripts to quickly load your data for training or evaluation.

## Requirements
Add the following to your `requirements.txt`:
```
datasets
```
Install with:
```
pip install -r requirements.txt
```

## Scripts

### 1. `load_hf_dataset.py`
Loads a dataset directly from Hugging Face Hub.
- By default, loads the full dataset (no subsetting).
- If you want to subset, uncomment and modify the relevant lines.
- Example usage:
```python
from datasets import load_dataset

dataset = load_dataset("taskydata/baize_chatbot")
print(dataset.column_names)
print(dataset['train'][0])
```

### 2. Loading Local or Colab Data
To load local data files (on your PC or in Colab), update the file paths in `data_files`:
```python
from datasets import load_dataset

data_files = {
    "train": "/path/to/your/train.jsonl",  # Use local path or Colab path
    "eval":  "/path/to/your/test.jsonl",
}
dataset = load_dataset("json", data_files=data_files)
```
- Just specify the correct file paths for your environment (local or Colab).
- No need for separate scripts; only the file path changes.

## Notes
- Make sure your file paths are correct for your environment (local PC, Colab, etc).
- The `datasets` library supports many formats (JSON, CSV, Parquet, etc).
- For large datasets, subsetting can be done using `.shuffle()` and `.select()` if needed.

For more details, see the official Hugging Face Datasets documentation: https://huggingface.co/docs/datasets
