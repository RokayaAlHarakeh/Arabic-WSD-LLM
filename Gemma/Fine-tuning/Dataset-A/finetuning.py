# ── 0. House-keeping ─────────────────────────────────────────────────────
import os, json, random, pathlib
from dotenv import load_dotenv
from datasets import Dataset

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# Load your HF token from .env and save to the HF cache
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")          # optional for public Unsloth weights (passed to from_pretrained)

# ── CONFIG (override via env vars so the SAME script serves both phases) ──
# Phase 1:  WSD_BASE_MODEL=unsloth/gemma-2-2b      WSD_MODEL_TAG=gemma2_2b
# Phase 2:  WSD_BASE_MODEL=unsloth/gemma-4-e4b-it  WSD_MODEL_TAG=gemma4_e4b
PROJECT_DIR = os.environ.get("WSD_PROJECT_DIR", "/content/drive/MyDrive/WSD_Project")
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
BASE_MODEL  = os.environ.get("WSD_BASE_MODEL", "unsloth/gemma-2-2b")
MODEL_TAG   = os.environ.get("WSD_MODEL_TAG", "gemma2_2b")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs", MODEL_TAG)
DATA_JSONL  = os.path.join(DATA_DIR, "fine_tuning_dataset_elrazzaz.jsonl")
PUSH_TO_HUB = os.environ.get("WSD_PUSH_TO_HUB", "0") == "1"

# ── 1. Model + LoRA setup (Unsloth style) ────────────────────────────────
MAX_SEQ_LEN     =  int(os.environ.get("WSD_MAX_SEQ_LEN", 1024))  # lower to 768 if a T4 OOMs on 4B
LOAD_IN_4BIT    = True          # memory-friendly
DTYPE           = None          # auto-detect (fp16 on T4/V100, bf16 on A100+)
print(f"▶ Base model: {BASE_MODEL}  |  tag: {MODEL_TAG}")
print(f"▶ Data: {DATA_JSONL}\n▶ Outputs: {OUTPUT_DIR}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = BASE_MODEL,
    max_seq_length   = MAX_SEQ_LEN,
    load_in_4bit     = LOAD_IN_4BIT,
    dtype            = DTYPE,
    token            = HF_TOKEN,       
)

model = FastLanguageModel.get_peft_model(
    model,
    r                       = 32,       # LoRA rank
    target_modules          = ["q_proj","k_proj","v_proj","o_proj",
                               "gate_proj","up_proj","down_proj"],
    lora_alpha              =32,
    lora_dropout            = 0.05,
    bias                    = "none",
    use_gradient_checkpointing = "unsloth",  # memory saver for long ctx
    random_state            = 3407,
    use_rslora              = False,
    loftq_config            = None,
)

# ── 2. Load & split JSONL ────────────────────────────────────────────────
def load_jsonl(path):
    return [json.loads(l) for l in pathlib.Path(path)
                               .read_text(encoding="utf-8").splitlines()]

records = load_jsonl(DATA_JSONL)
random.seed(42); random.shuffle(records)
cut = int(len(records) * 0.9)
train_records, eval_records = records[:cut], records[cut:]

# ── 3. Prompt formatting (Alpaca style) ──────────────────────────────────
alpaca_prompt = """Below is an instruction that describes a task, \
paired with an input that provides further context. \
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS = tokenizer.eos_token

def add_text_column(batch):
    return {
        "text": [
            alpaca_prompt.format(i, inp, out) + EOS
            for i, inp, out in zip(batch["instruction"],
                                   batch["input"],
                                   batch["output"])
        ]
    }

train_ds = (Dataset.from_list(train_records)
            .map(add_text_column, batched=True, remove_columns=[]))
eval_ds  = (Dataset.from_list(eval_records )
            .map(add_text_column, batched=True, remove_columns=[]))

print(f"✅ Train={len(train_ds)}  Eval={len(eval_ds)}")
print("🔎 Sample formatted text:\n", train_ds[0]["text"][:500], "...")

# ── 4. TrainingArguments + SFTTrainer ────────────────────────────────────
training_args = TrainingArguments(
    output_dir                 = os.path.join(OUTPUT_DIR, "checkpoints"),
    per_device_train_batch_size= 1,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps= 8,
    num_train_epochs=3,
    warmup_steps               = 50,
    learning_rate              = 2e-4,
    fp16                       = not is_bfloat16_supported(),
    bf16                       = is_bfloat16_supported(),
    logging_steps              = 20,
    eval_strategy              = "steps",
    eval_steps                 = int(os.environ.get("WSD_EVAL_STEPS", 100)),  # raise to 500 to spend less time on eval
    save_total_limit           = 2,
    optim                      = "adamw_8bit",
    weight_decay               = 0.01,
    lr_scheduler_type          = "linear",
    seed                       = 3407,
    report_to                  = "none",
)

trainer = SFTTrainer(
    model               = model,
    tokenizer           = tokenizer,
    train_dataset       = train_ds,
    eval_dataset        = eval_ds,
    dataset_text_field  = "text",
    max_seq_length      = MAX_SEQ_LEN,
    dataset_num_proc    = 4,
    packing             = True,   # turn on for many short examples
    args                = training_args,
)

# ── 5. Train ─────────────────────────────────────────────────────────────
trainer.train()

# ── 6. Save merged 16-bit weights (this is what infer_model.py loads) ─────
MERGED_DIR = os.path.join(OUTPUT_DIR, "merged_16bit")
model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
print(f"💾 Merged model saved to {MERGED_DIR}")

# Optional: push to the Hub. Set WSD_PUSH_TO_HUB=1 and WSD_HUB_REPO=<user>/<repo>.
if PUSH_TO_HUB:
    repo_id = os.environ["WSD_HUB_REPO"]
    model.push_to_hub_merged(
        repo_id     = repo_id,
        tokenizer   = tokenizer,
        save_method = "merged_16bit",
        token       = HF_TOKEN,
    )
    print(f"🚀 Pushed merged model to the Hub: {repo_id}")

print("✅ Finetuning complete.")