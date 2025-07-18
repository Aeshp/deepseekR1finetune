# Fine-tune Checkpoint Notebooks

This folder contains Jupyter notebooks for further fine-tuning the model `Aeshp/deepseekR1tunedchat`, which is already trained on customer service and general chat datasets. You can use these notebooks to continue training the model on your own domain-specific data.

## Overview
- The base model (`Aeshp/deepseekR1tunedchat`) has been trained and pushed to Hugging Face Hub.
- You can load this checkpoint and fine-tune it further on your own dataset (e.g., medical, legal, education, etc.).
- Example notebooks are provided for training and for adjusting hyperparameters.

## Notebooks

### 1. `Aeshp_deepseekR1tunedchat.ipynb`
- Contains code to load the checkpointed model from Hugging Face Hub and further fine-tune it.
- Shows how to set up your dataset, configure training, and run further fine-tuning.
- Includes instructions for handling bitsandbytes errors and environment setup.
- Demonstrates pushing updated weights to Hugging Face Hub after training.
- Uses approximately 80 examples for testing and 500 for training as a demonstration. For better performance, use a larger training and testing dataset.
- Make sure you install all required packages and libraries before running the notebook (see the first cells for installation commands).

### 2. `hyperparams.ipynb`
- Provides example code for setting and adjusting hyperparameters for training.
- Shows how to calculate steps, epochs, batch size, and learning rate for small datasets (e.g., 500 examples).
- You can modify these values to fit your own dataset size and hardware.

## How to Use
1. **Install required packages:**
   - Run the installation cells at the top of the notebook to install all necessary libraries.
2. **Prepare your dataset:**
   - Format your data as JSONL or another supported format and update the paths in the notebook.
3. **Open `Aeshp_deepseekR1tunedchat.ipynb`:**
   - Follow the steps to load the model, set up your data, and start training.
   - Adjust hyperparameters as needed using `hyperparams.ipynb`.
4. **Monitor training:**
   - Use TensorBoard for logging and monitoring if desired.
5. **Push your updated model:**
   - After training, push your new weights to Hugging Face Hub for sharing or deployment.

## Notes
- The example uses only 500 training examples and about 80 for testing for demonstration. For real use, update the dataset and hyperparameters accordingly for better results.
- You can further fine-tune the checkpointed model on any domain-specific dataset.
- For troubleshooting bitsandbytes or other library issues, see the notebook cells for installation tips.

For more details, see the Hugging Face documentation: https://huggingface.co/docs/hub/index
