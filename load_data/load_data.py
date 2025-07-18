from datasets import load_dataset

# Define the data files dictionary
data_files = {
    "train": "/content/train.jsonl", #path to your data
    "eval":  "/content/test.jsonl",
}

# load the datasets
dataset = load_dataset("json", data_files=data_files)

# Print the column names of the training dataset
print("Columns:", dataset["train"].column_names)


dataset['train'][0]
dataset['eval'][0]