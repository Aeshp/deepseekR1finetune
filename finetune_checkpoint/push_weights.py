# it'll push your fine tuned weights..
import torch
from unsloth import FastLanguageModel
from huggingface_hub import login

# login to Hugging Face
login() #need token

# Replace with your trained model path - this is the model you want to push
model_path = "outputs/checkpoints"  

# Replace "your_username/your_model_name" with your Hugging Face username and model repository name
repo_name = "your_username/your_model_name"  

# Load the model and tokenizer
print(f"Loading model from {model_path}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,  # or use "Aeshp/deepseekfinetuned01" if pushing an existing HF model
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="auto",
)

# push the model and tokenizer to the Hugging Face Hub
print(f"Pushing model and tokenizer to {repo_name}...")
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"Model and tokenizer successfully pushed to https://huggingface.co/{repo_name}")