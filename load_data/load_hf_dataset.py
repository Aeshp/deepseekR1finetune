from datasets import load_dataset

dataset = load_dataset("taskydata/baize_chatbot")

# Load the full train split (no subsetting)
#if subsettingneeded then split the train set to exactly 10,000 examples and set N
print(dataset.column_names)
print(dataset['train'][0])