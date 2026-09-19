import torch
from unsloth import FastLanguageModel

MODEL_NAME = "google/gemma-4-12B"
MAX_SEQ_LENGTH = 4096

print("Testing Unsloth model loading...")
print("Model:", MODEL_NAME)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

print("Model loaded successfully.")
print("Tokenizer loaded successfully.")
print("EOS token:", tokenizer.eos_token, tokenizer.eos_token_id)
print("PAD token:", tokenizer.pad_token, tokenizer.pad_token_id)

#Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
#  Loading weights: 100%|██████████| 677/677 [01:07<00:00, 10.06it/s] Model loaded successfully. 
# Tokenizer loaded successfully. EOS token: <eos> 1 PAD token: <pad> 0 ok next