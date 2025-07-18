import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer


model_name = "YourModelNameHere"  # Replace with your tuned model

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,  
    load_in_4bit=True,
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


instruction = "Answer my questions"
input_text = "Hiii How are You ?"


prompt = alpaca_prompt.format(
    instruction,
    input_text,
    "",  
)

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

text_streamer = TextStreamer(tokenizer)


_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)