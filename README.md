# DeepSeekR1 Fine-Tuning and Inference Codebase

This repository contains code for fine-tuning the DeepSeek-R1-Distill-Llama-8B model using Unsloth, LoRA, and Hugging Face tools. The model has been fine-tuned on customer service and general chat datasets and is open-sourced for further fine-tuning on domain-specific datasets.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Aeshp/deepseekR1tunedchat)

## 📋 Overview

I've taken the open-source model [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) (loaded via Unsloth in 4-bit as [unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit)) and fine-tuned it on several datasets:

- [taskydata/baize_chatbot](https://huggingface.co/datasets/taskydata/baize_chatbot)
- [MohammadOthman/mo-customer-support-tweets-945k](https://huggingface.co/datasets/MohammadOthman/mo-customer-support-tweets-945k)
- [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

The training was done in multiple steps. After fine-tuning, I merged the trained weights with the base model and pushed the complete model to my Hugging Face model repository: [Aeshp/deepseekR1tunedchat](https://huggingface.co/Aeshp/deepseekR1tunedchat).

## ⚠️ Important Notes

- The model is released under the MIT license, allowing free use in applications and further fine-tuning
- This model may sometimes hallucinate, as is common with language models
- For effective fine-tuning, users should have a sufficient amount of high-quality data to avoid overfitting

## 🗂️ Repository Structure

- `Example_dataset/`: Sample datasets for testing and demonstration
- `Finetune_base/`: Python scripts for fine-tuning the base model
- `fineTune_nb_base/`: Jupyter notebooks for fine-tuning the base model
- `finetune_checkpoint/`: Python scripts for fine-tuning from the checkpoint model
- `finetune_checkpoint_nb/`: Jupyter notebooks for fine-tuning from the checkpoint model
- `huggingface_push/`: Scripts for pushing models to Hugging Face Hub
- `load_data/`: Utilities for loading and preprocessing datasets
- `ModelCall/`: Scripts for model inference and deployment

## 🚀 Getting Started

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Aeshp/deepseekR1finetune.git
   cd deepseekR1finetune
   ```

2. Choose your workflow:
   - For base model fine-tuning: `cd Finetune_base`
   - For checkpoint model fine-tuning: `cd finetune_checkpoint`
   - For notebooks (Jupyter/Colab): Use the corresponding notebook directories

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

### Fine-tuning Workflows

#### Base Model Fine-tuning

```bash
cd Finetune_base
python unsloth_deepseek_basetune.py
```

#### Checkpoint Model Fine-tuning

```bash
cd finetune_checkpoint
python deepseekR1tunedchat.py
```

### Using TensorBoard for Monitoring

During training, you can monitor progress using TensorBoard:

```bash
tensorboard --logdir outputs/runs/
```

Then open http://localhost:6006 in your browser to view training metrics.

### Dataset Creation

To create your own dataset for fine-tuning, follow this format:

```json
{
  "instruction": "Your instruction here",
  "input": "Additional context (optional)",
  "output": "The expected model output",
  "text": "Combined prompt + response template"
}
```

Convert your data to JSONL format and split it into train and test sets.

### Running Inference

After fine-tuning, run inference with:

```bash
cd finetune_checkpoint
python inferance.py
```

You can modify the prompt and parameters in the script to suit your needs.

### Pushing to Hugging Face Hub

#### Push Only Weights

```bash
cd huggingface_push
python push_only_weights.py
```

#### Push Merged Model with Base Weights

```bash
cd huggingface_push
python merge_weights_and_push.py
```

Remember to update the repository name and login with your Hugging Face token.

## 📊 Model Details

```
base_model:
- deepseek-ai/DeepSeek-R1-Distill-Llama-8B
Quantized:
- unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit
tags:
- text-generation-inference
- transformers
- unsloth
- llama
- bitsandbytes
- 8B
license: mit
language:
- en
datasets:
- taskydata/baize_chatbot
- MohammadOthman/mo-customer-support-tweets-945k
- bitext/Bitext-customer-support-llm-chatbot-training-dataset
new_version: Aeshp/deepseekR1tunedchat
pipeline_tag: text-generation
library_name: transformers
```

This model is fine-tuned on customer service and chatbot data and is ready to be fine-tuned on any specific domain dataset. To learn how to tune it further, refer to this GitHub repository: [Aeshp/deepseekR1finetune](https://github.com/Aeshp/deepseekR1finetune).

The model uses the base model [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) and is loaded in 4-bit as [unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit). It has been tuned on the datasets mentioned above and pushed with merged weights to the base model.

## 📚 References

### Hugging Face Models
- [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit)

### GitHub Repositories
- [meta-llama/llama](https://github.com/meta-llama/llama)
- [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)

### Papers
- [DeepSeek R1 Paper](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
- [Unsloth Documentation](https://docs.unsloth.ai/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.