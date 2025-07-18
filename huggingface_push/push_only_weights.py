from huggingface_hub import login

# use login() for both script and notebook compatibility
login()  #need hf token

# Replace "your_username/your_model_name" with your Hugging Face username and model repository name
repo_name = "your_username/your_model_name"

# push the model and tokenizer to the Hugging Face Hub
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"Model and tokenizer pushed to https://huggingface.co/{repo_name}")