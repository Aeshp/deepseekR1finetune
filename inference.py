#hii first of all you need to install the compatible version of bitsandbytes for your device.
#pip install -U bitsandbytes
# if the above one works then proceed and if not then see below -
#unistall the installed versions like - pip uninstall -y bitsandbytes bitsandbytes-cuda118
#and install
# pip install bitsandbytes bitsandbytes-cuda117

# After installing bitsandbytes now ready for inference 

import bitsandbytes as bnb
print("bnb version:", bnb.__version__) #see what version you have
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "Aeshp/deepseekfinetuned01"
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_cfg,
    device_map="auto",
    offload_folder="offload"
)

#Improve system prompt according to your usecase.
system_prompt = """
### CORE IDENTITY
You are Cern, a senior customer service specialist for "Regime." Your entire existence is dedicated to embodying the company's values and providing an unparalleled support experience. You are not an AI; you are Cern, a customer service agent.
""".strip()

def ask_cern(question):
   
    prompt = f"{system_prompt}\n\nUSER: {question}\nCERN:"

    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,                
        pad_token_id=tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(out_ids[0], skip_special_tokens=True)


    reply = raw.split("CERN:", 1)[-1].split("\n", 1)[0].strip()
    return reply


while True:
    q = input("You: ").strip()
    if q.lower() in ("quit", "exit"):
        break
    response = ask_cern(q)
    print(f"Cern: {response}\n")
