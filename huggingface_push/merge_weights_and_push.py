from peft import PeftModel
from huggingface_hub import login, Repository, HfApi

# log in 
login()  # need token

# merge LoRA adapters into the base model via PEFT
merged_model = model.merge_and_unload()

save_dir = "merged_model"
merged_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# push via the Hub API (avoiding any Unsloth save path)
api = HfApi()
api.upload_folder(
    folder_path = save_dir,
    repo_id     = "Yourusername/modelrepo",
    repo_type   = "model",
    token       = True  # uses your login token
)

print("Full merged model pushed to the Hub successfully!")
