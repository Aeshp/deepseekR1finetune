# Example Dataset: IITG Professors Contact Details

This repository contains a **demo dataset** of contact details for professors in the Online B.Sc. Hons. Data Science & Artificial Intelligence program at the Indian Institute of Technology Guwahati (IITG). All information is publicly available online. Use this data **only** for demonstration and testing purposes (e.g., fine‑tuning small models, running smoke tests, or proof‑of‑concept experiments).

---

## Dataset Format

The dataset is stored in **JSONL** format. Each line is a JSON object with the following fields:

* **`instruction`** (string): The question or task prompt, e.g., "Who teaches DA108 in the ... program at IITG?"
* **`input`** (string): Additional context or input (empty if none).
* **`output`** (string): The correct answer, e.g., "DA108 is taught by Neeraj Sir."
* **`text`** (string): A combined prompt + response template, formatted for direct use in chat‑style fine‑tuning:

  ```text
  Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  {instruction}

  ### Response:
  {output}
  ```

### Example Records

```jsonl
{  
  "instruction": "Who teaches DA108 in the Online BSc Hons, Data Science and Artificial Intelligence program at the Indian Institute of Technology Guwahati (IITG)?",  
  "input": "",  
  "output": "DA108 is taught by Neeraj Sir.",  
  "text": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWho teaches DA108 in the Online BSc Hons, Data Science and Artificial Intelligence program at the Indian Institute of Technology Guwahati (IITG)?\n\n### Response:\nDA108 is taught by Neeraj Sir."
}
{  
  "instruction": "Which course is taught by Debanga sir in the Online BSc Hons, Data Science and Artificial Intelligence program at the Indian Institute of Technology Guwahati (IITG)?",  
  "input": "",  
  "output": "DA107 is taught by Debanga sir.",  
  "text": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhich course is taught by Debanga sir in the Online BSc Hons, Data Science and Artificial Intelligence program at the Indian Institute of Technology Guwahati (IITG)?\n\n### Response:\nDA107 is taught by Debanga sir."
}
```

---

## Included Files

* **`train.jsonl`** — 400 examples for training.
* **`test.jsonl`**  — 100 examples for evaluation.

## Usage

1. **Install dependencies** (adjust as necessary):

   ```bash
   pip install -r requirements.txt
   ```

2. **Fine‑tune** on the demo dataset:

   ```bash
   python scripts/finetune_continue.py \
     --train_file data/train.jsonl \
     --eval_file  data/test.jsonl \
     --model_name Aeshp/deepseekR1tunedchat \
     --output_dir outputs/demo_finetune
   ```

3. **Evaluate** on the test split:

   ```bash
   python scripts/finetune_continue.py \
     --predict_only \  
     --eval_file data/test.jsonl \
     --model_dir outputs/demo_finetune
   ```

> ⚠️ **Disclaimer:** All professor contact details are taken from publicly accessible IITG web pages. Use this data responsibly and for demonstration purposes only.

---

© 2025 Aesh P. — MIT License
