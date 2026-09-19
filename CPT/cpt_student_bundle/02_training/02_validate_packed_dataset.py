import json
from pathlib import Path
from datasets import load_dataset

TRAIN_FILE = "data_cpt/final_cpt_v1/packed/train_cleanheaders_packed_4096.jsonl"
VAL_FILE = "data_cpt/final_cpt_v1/packed/val_cleanheaders_packed_4096.jsonl"

print("Checking packed dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
    },
)

print(dataset)

sample = dataset["train"][0]

print("Columns:", dataset["train"].column_names)
print("First sample keys:", sample.keys())
print("input_ids length:", len(sample["input_ids"]))
print("attention_mask length:", len(sample["attention_mask"]))
print("First 20 input_ids:", sample["input_ids"][:20])
print("First 20 attention_mask:", sample["attention_mask"][:20])

assert "input_ids" in sample
assert "attention_mask" in sample
assert len(sample["input_ids"]) == 4096
assert len(sample["attention_mask"]) == 4096

print("Packed dataset OK.")