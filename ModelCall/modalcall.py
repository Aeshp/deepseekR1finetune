import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

model_name = "yourusername/modelname"  #Use your fine-tuned model name here
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,  # Auto-detects the data type
    load_in_4bit=True, # Use 4-bit quantization for efficiency
)


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# add your instruction and input for your prompt
instruction = "Solve my query"
input_text = "I have purchased a new laptop and now it is occuring issue so what to do ?"

prompt = alpaca_prompt.format(
    instruction,
    input_text,
    "",  
)


inputs = tokenizer([prompt], return_tensors="pt").to("cuda")


text_streamer = TextStreamer(tokenizer)


# generate the response
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)