# ── 0. House-keeping ─────────────────────────────────────────────────────
# Plain transformers + PEFT QLoRA — NO Unsloth on purpose. The Colab
# `pip install -U` stack (unsloth 2026.6.6 / transformers 5.5.0) has a broken
# Gemma2 forward pass: training through it produced a corrupt adapter that
# generated garbage on a correct forward (see RUNBOOK + memory). The base model
# is fine under plain transformers, so we train QLoRA the standard way and never
# import unsloth.
import os, json, random, pathlib, glob, shutil
import torch
from dotenv import load_dotenv
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")          # optional for public weights

# ── CONFIG (override via env vars so the SAME script serves both phases) ──
# Phase 1:  WSD_BASE_MODEL=unsloth/gemma-2-2b      WSD_MODEL_TAG=gemma2_2b
# Phase 2:  WSD_BASE_MODEL=unsloth/gemma-4-e4b-it  WSD_MODEL_TAG=gemma4_e4b
PROJECT_DIR = os.environ.get("WSD_PROJECT_DIR", "/content/drive/MyDrive/WSD_Project")
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
BASE_MODEL  = os.environ.get("WSD_BASE_MODEL", "unsloth/gemma-2-2b")
MODEL_TAG   = os.environ.get("WSD_MODEL_TAG", "gemma2_2b")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs", MODEL_TAG)
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "adapter")          # what infer_model.py loads
DATA_JSONL  = os.path.join(DATA_DIR, "fine_tuning_dataset_elrazzaz.jsonl")
PUSH_TO_HUB = os.environ.get("WSD_PUSH_TO_HUB", "0") == "1"

MAX_SEQ_LEN = int(os.environ.get("WSD_MAX_SEQ_LEN", 1024))  # lower to 768 if a T4 OOMs
MAX_STEPS   = int(os.environ.get("WSD_MAX_STEPS", 0))       # >0 = quick smoke run

BF16 = torch.cuda.is_bf16_supported()           # T4 -> False (fp16); A100 -> True
COMPUTE_DTYPE = torch.bfloat16 if BF16 else torch.float16

print(f"▶ Base model: {BASE_MODEL}  |  tag: {MODEL_TAG}")
print(f"▶ Data: {DATA_JSONL}\n▶ Outputs: {OUTPUT_DIR}  (adapter -> {ADAPTER_DIR})")

# ── 1. Tokenizer + 4-bit base + LoRA (standard QLoRA) ────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype    = COMPUTE_DTYPE,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config = bnb_config,
    dtype               = COMPUTE_DTYPE,
    attn_implementation = "sdpa",       # sdpa handles Gemma2 softcapping correctly here and is ~2x faster than eager
    token               = HF_TOKEN,
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r              = 32,
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = "CAUSAL_LM",
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_config)
model.config.use_cache = False                  # required with gradient checkpointing
model.print_trainable_parameters()

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

train_ds = Dataset.from_list(train_records).map(add_text_column, batched=True)
eval_ds  = Dataset.from_list(eval_records ).map(add_text_column, batched=True)

print(f"✅ Train={len(train_ds)}  Eval={len(eval_ds)}")
print("🔎 Sample formatted text:\n", train_ds[0]["text"][:500], "...")

# Tokenize (causal-LM; the collator builds labels from input_ids).
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN)

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(tokenize_fn,  batched=True, remove_columns=eval_ds.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ── 4. TrainingArguments + Trainer ───────────────────────────────────────
training_kwargs = dict(
    output_dir                  = os.path.join(OUTPUT_DIR, "checkpoints"),
    per_device_train_batch_size = 1,
    per_device_eval_batch_size  = 1,
    gradient_accumulation_steps = 8,
    warmup_steps                = 50,
    learning_rate               = 2e-4,
    fp16                        = not BF16,
    bf16                        = BF16,
    logging_steps               = 20,
    save_strategy               = "steps",
    save_steps                  = int(os.environ.get("WSD_SAVE_STEPS", 500)),  # resumable ckpts on Drive
    save_total_limit            = 2,
    optim                       = "adamw_8bit",
    weight_decay                = 0.01,
    lr_scheduler_type           = "linear",
    seed                        = 3407,
    gradient_checkpointing      = True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    report_to                   = "none",
)
if MAX_STEPS > 0:
    training_kwargs["max_steps"] = MAX_STEPS    # quick smoke run to validate the stack
    print(f"⚠️ WSD_MAX_STEPS={MAX_STEPS} — SHORT smoke run, NOT full training")
else:
    training_kwargs["num_train_epochs"] = float(os.environ.get("WSD_EPOCHS", 3))

# Eval is OFF by default: eval_loss isn't used for selection here and the pass is
# slow (~12 min over the dev split). Set WSD_EVAL_STEPS>0 to re-enable.
_eval_steps = int(os.environ.get("WSD_EVAL_STEPS", 0))
if _eval_steps > 0:
    training_kwargs.update(eval_strategy="steps", eval_steps=_eval_steps)
else:
    training_kwargs["eval_strategy"] = "no"
    print("ℹ️ eval disabled (set WSD_EVAL_STEPS>0 to re-enable)")

training_args = TrainingArguments(**training_kwargs)

trainer = Trainer(
    model            = model,
    args             = training_args,
    train_dataset    = train_ds,
    eval_dataset     = eval_ds,
    data_collator    = data_collator,
    processing_class = tokenizer,
)

# ── 5. Train (auto-resume from the last *complete* checkpoint on Drive) ───
# A checkpoint from a crash/disconnect mid-save lacks trainer_state.json and
# would break resume, so drop those first.
_ckpt_root = os.path.join(OUTPUT_DIR, "checkpoints")
for _d in glob.glob(os.path.join(_ckpt_root, "checkpoint-*")):
    if not os.path.isfile(os.path.join(_d, "trainer_state.json")):
        print(f"⚠️ Removing incomplete checkpoint: {_d}")
        shutil.rmtree(_d, ignore_errors=True)
_valid = sorted(
    glob.glob(os.path.join(_ckpt_root, "checkpoint-*")),
    key=lambda p: int(p.rsplit("-", 1)[-1]),
)
_resume = _valid[-1] if _valid else False
print(f"▶ Resuming from {_resume}" if _resume else "▶ Starting fresh")
trainer.train(resume_from_checkpoint=_resume)

# ── 6. Save the LoRA adapter (this is what infer_model.py loads) ──────────
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"💾 Adapter saved to {ADAPTER_DIR}")

# Optional: push the adapter to the Hub. Set WSD_PUSH_TO_HUB=1 and WSD_HUB_REPO.
if PUSH_TO_HUB:
    repo_id = os.environ["WSD_HUB_REPO"]
    model.push_to_hub(repo_id, token=HF_TOKEN)
    tokenizer.push_to_hub(repo_id, token=HF_TOKEN)
    print(f"🚀 Pushed adapter to the Hub: {repo_id}")

print("✅ Finetuning complete.")
