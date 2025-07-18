import math
import datetime
import os
from datasets import load_dataset
from transformers import TrainingArguments
from transformers.integrations import TensorBoardCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported, to_sharegpt, standardize_sharegpt, apply_chat_template
import matplotlib.pyplot as plt

# load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "Aeshp/deepseekR1tunedchat",  # this is tuning again on top of base model
    max_seq_length = 2048,
    dtype          = None,
    load_in_4bit   = True,
    device_map     = "auto",
)

# apply LoRA/PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r                  = 4,
    target_modules     = ["q_proj","k_proj","v_proj","o_proj"],
    lora_alpha         = 16,
    lora_dropout       = 0,
    bias               = "none",
    use_gradient_checkpointing = "unsloth",
    random_state       = 42,
    use_rslora         = False,
    loftq_config       = None,
)

# load your dataset
# Update the paths to your actual train/eval files
data_files = {
    "train": "./train.jsonl",
    "eval":  "./test.jsonl",
}
dataset = load_dataset("json", data_files=data_files)

print("Columns:", dataset["train"].column_names)
print("First train example:", dataset["train"][0])
print("First eval example:", dataset["eval"][0])

# Training hyperparameters
N_train        = 500      # adjust according to your dataset size
batch_size     = 1
grad_accum     = 8
num_epochs     = 4
steps_per_epoch= math.ceil(N_train / (batch_size * grad_accum))
total_steps    = num_epochs * steps_per_epoch
warmup_steps   = 10
learning_rate  = 2e-5

print(f"N_train: {N_train}")
print(f"batch_size: {batch_size}")
print(f"grad_accum: {grad_accum}")
print(f"num_epochs: {num_epochs}")
print(f"steps_per_epoch: {steps_per_epoch}")
print(f"total_steps: {total_steps}")
print(f"warmup_steps: {warmup_steps}")
print(f"learning_rate: {learning_rate}")

# tensorBoard and output directories
output_dir = "outputs"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_name_for_log = "Aeshp_deepseekR1tunedchat"
tensorboard_log_dir = os.path.join(output_dir, "runs", f"{timestamp}_{model_name_for_log}")
checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(tensorboard_log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"TensorBoard log directory created: {tensorboard_log_dir}")
print(f"Checkpoint directory created: {checkpoint_dir}")

# setup SFTTrainer
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoardCallback(log_dir)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset["train"],
    eval_dataset = dataset["eval"],
    args = TrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum,
        warmup_steps = warmup_steps,
        num_train_epochs = num_epochs,
        learning_rate = learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1, # Log every step
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        save_strategy = "epoch",       # save checkpoint at the end of each epoch
    ),
    #callbacks=[tensorboard_callback], # Uncomment to enable TensorBoard callback
)

# train
trainer.train()

# evaluate
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)