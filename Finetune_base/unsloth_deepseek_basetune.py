import os
import datetime
import math
import torch
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel, is_bfloat16_supported
from peft import PeftModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from transformers.integrations import TensorBoardCallback

# hey load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit", #eg model_name
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    device_map = "auto",
)

# apply pEFT (LoRA)
model = FastLanguageModel.get_peft_model(
    model,
    r = 4,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
    use_rslora = True,
    loftq_config = None,
)

# Load dataset
print("Loading dataset...")
dataset = load_dataset("taskydata/baize_chatbot") #eg dataset , replace with your dataset
print("Columns:", dataset.column_names)
print("First train example:", dataset['train'][0])

# format dataset
chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.\n\n### Instruction:\n{INPUT}\n\n### Response:\n{OUTPUT}"""

def format_dataset_example(example):
    parts = example['input'].split('[|AI|]', 1)
    if len(parts) > 1:
        human_part = parts[0].replace('The conversation between human and AI assistant.\n', '').strip()
        if human_part.endswith('[|Human|]'):
            human_part = human_part[:-len('[|Human|]')].strip()
        ai_part = parts[1].strip()
        formatted_text = chat_template.format(INPUT=human_part, OUTPUT=ai_part)
        example['text'] = formatted_text
    else:
        example['text'] = ""
    return example

dataset["train"] = dataset["train"].map(format_dataset_example, num_proc=2, remove_columns=['topic', 'input'])
print("First formatted example:", dataset['train'][0]['text'])
print("Dataset column names after formatting:", dataset.column_names)

# tensorBoard and checkpoint setup
output_dir = "outputs"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_name_for_log = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
tensorboard_log_dir = os.path.join(output_dir, "runs", f"{timestamp}_{model_name_for_log}")
checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(tensorboard_log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"TensorBoard log directory created: {tensorboard_log_dir}")
print(f"Checkpoint directory created: {checkpoint_dir}")

# training hyperparameters
per_device_batch_size = 2 # adjust based on VRAM
grad_accum = 16 # if neded then effective batch size = 32
num_epochs = 1
max_steps = math.ceil((len(dataset["train"]) / (per_device_batch_size * grad_accum)) * num_epochs)

trainer = SFTTrainer(
    model            = model,
    tokenizer        = tokenizer,
    train_dataset    = dataset["train"],
    # eval_dataset   = dataset["test"], # Add this for evaluation
    # formatting_func= formatting_func,
    max_seq_length   = 2048,
    packing          = True,
    args = TrainingArguments(
        per_device_train_batch_size = per_device_batch_size,
        gradient_accumulation_steps = grad_accum,
        warmup_ratio                = 0.05,
        max_steps                   = max_steps,
        learning_rate               = 2e-5,
        fp16                        = not is_bfloat16_supported(),
        bf16                        = is_bfloat16_supported(),
        logging_steps               = 25,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "cosine",
        seed                        = 3407,
        output_dir                  = output_dir,
        report_to                   = "tensorboard",
        logging_dir                 = tensorboard_log_dir,
        logging_strategy            = "steps",
        save_strategy               = "steps",
        save_steps                  = 500,
        # evaluation_strategy       = "steps",
        # eval_steps                = 500,
    ),
)

# start training
trainer.args.num_train_epochs = num_epochs
trainer_stats = trainer.train()

# now go for inference example and ask questions about the dataset.
