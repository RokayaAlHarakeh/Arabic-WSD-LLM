"""
Full continued-pretraining (CPT) run for Gemma 2-2B on the packed Lebanese-legal
dataset, 4-bit QLoRA.

Adapted from 05_full_training_ORIGINAL.py (Gemma 4 12B, Unsloth). Everything
downstream of the model loader is unchanged from the original and is stock
transformers: packed_data_collator, validate_example, strip_non_training_columns,
the resume logic, load_best_model_at_end, and run_summary.json.

CHANGES FROM THE ORIGINAL
-------------------------
1. HF cache env defaults removed. The original pinned them to
   /opt/dlami/nvme/huggingface (an AWS DLAMI path) which does not exist on Colab.
2. boto3 is optional -- it was imported at module level and fails on Colab.
3. Defaults: google/gemma-2-2b, max_seq_length 2048, Colab/Drive paths.
4. Loader is selectable (--loader), because Gemma 2 is text-only and returns a
   TOKENIZER where Gemma 4 returned a PROCESSOR, and because of (6) below.
5. Precision is auto-detected. The original hardcoded bf16=True, which fails on a
   T4 (Turing, no bf16).
6. An initial-loss sanity probe runs before training. See below -- this is the
   most important addition.

THE SANITY PROBE
----------------
This repo has already lost one full training run to a silently broken Unsloth
patch (unsloth 2026.6.6 + transformers 5.5.0 on Gemma 2): the forward pass was
corrupt, training started at loss ~25 against a random baseline of ~12.5, and the
resulting adapter was unusable. It did not raise -- it just produced garbage.

So a try/except around the loader cannot catch it. Instead, before training
starts, we run one forward pass and compare the loss to ln(vocab_size), which is
the loss of a uniform-random model. A correctly loaded pretrained model must be
well below that number. If it is above, the forward pass is broken and we abort
rather than burn a GPU session producing another unusable adapter.

Run:
    python 05_full_training_gemma2_2b.py --skip_s3_upload

Smoke test first (always):
    python 05_full_training_gemma2_2b.py --dry_run_samples 64 --num_epochs 1 --skip_s3_upload

Resume is automatic: re-running the same command picks up the latest checkpoint.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset

from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

# boto3 is only needed for the S3 backup path, which is unused on Colab.
try:
    import boto3
except ImportError:
    boto3 = None

# peft on the current Colab stack hard-errors via is_torchao_available() on the
# old torchao Colab preinstalls, even though we never use torchao (bnb only).
# Same guard as Gemma/Fine-tuning/Dataset-A/finetuning.py.
try:
    import peft.import_utils as _pi, peft.tuners.lora.torchao as _pt
    _pi.is_torchao_available = _pt.is_torchao_available = lambda: False
except Exception:
    pass


# -----------------------------------------------------------------------------
# Defaults (override any of these on the command line)
# -----------------------------------------------------------------------------
MODEL_NAME = "google/gemma-2-2b"

# RunPod: everything must live under /workspace (the network volume). The container
# disk is destroyed when the pod is terminated.
WORKSPACE = os.environ.get("WSD_WORKSPACE", "/workspace")

TRAIN_FILE = f"{WORKSPACE}/packed_2048/train_packed_2048.jsonl"
VAL_FILE = f"{WORKSPACE}/packed_2048/val_packed_2048.jsonl"

OUTPUT_DIR = f"{WORKSPACE}/adapters/gemma2_2b_cpt_v1"

MAX_SEQ_LENGTH = 2048

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# -----------------------------------------------------------------------------
# Data collator and validation (unchanged from the original)
# -----------------------------------------------------------------------------
def packed_data_collator(features):
    """
    Collator for fixed-length packed CPT examples.

    The JSONL records already contain input_ids and attention_mask, all of
    length MAX_SEQ_LENGTH, so no tokenizer padding is needed. Labels are
    copied from input_ids; padding positions (attention_mask == 0) are
    masked with -100. Real EOS separators keep attention_mask == 1, so the
    model still learns document boundaries.
    """
    input_ids = torch.tensor(
        [example["input_ids"] for example in features],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [example["attention_mask"] for example in features],
        dtype=torch.long,
    )

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def validate_example(example, split_name, index, max_seq_length):
    required_columns = {"input_ids", "attention_mask"}
    missing_columns = required_columns.difference(example.keys())

    if missing_columns:
        raise ValueError(
            f"{split_name} row {index} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    input_length = len(example["input_ids"])
    mask_length = len(example["attention_mask"])

    if input_length != mask_length:
        raise ValueError(
            f"{split_name} row {index} has mismatched lengths: "
            f"input_ids={input_length}, attention_mask={mask_length}"
        )

    if input_length != max_seq_length:
        raise ValueError(
            f"{split_name} row {index} has length {input_length}, "
            f"expected {max_seq_length}"
        )


def strip_non_training_columns(dataset, split_name):
    keep_columns = {"input_ids", "attention_mask"}
    columns_to_remove = [
        column
        for column in dataset.column_names
        if column not in keep_columns
    ]

    if columns_to_remove:
        print(f"Removing non-training {split_name} columns:", columns_to_remove)
        dataset = dataset.remove_columns(columns_to_remove)

    print(f"Trainer {split_name} columns:", dataset.column_names)
    return dataset


# -----------------------------------------------------------------------------
# Model loading -- two paths, see module docstring
# -----------------------------------------------------------------------------
def load_with_unsloth(args):
    """The original path. Returns (model, tokenizer)."""
    from unsloth import FastLanguageModel  # noqa: F401  (must precede transformers use)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=TARGET_MODULES,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    return model, tokenizer


def load_with_transformers(args, compute_dtype):
    """
    Plain transformers QLoRA -- the configuration already validated in this repo
    for Gemma 2 (Gemma/Fine-tuning/Dataset-A/finetuning.py). Returns (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        # sdpa handles Gemma 2's logit soft-capping correctly here and is ~2x
        # faster than eager. Matched to the SFT run for comparability.
        attn_implementation="sdpa",
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    return model, tokenizer


def initial_loss_probe(model, dataset, max_rows=1):
    """
    One forward pass before training, compared against the uniform-random loss
    ln(vocab_size).

    A correctly loaded pretrained model scores far below random on in-domain
    text. A loss at or above random means the forward pass is broken -- which is
    exactly how the earlier Unsloth/Gemma-2 failure in this repo presented, and
    it did not raise an exception. Aborting here costs seconds; not aborting
    costs a GPU session and produces an unusable adapter.
    """
    vocab_size = model.config.vocab_size
    random_loss = math.log(vocab_size)

    rows = [dataset[i] for i in range(min(max_rows, len(dataset)))]
    batch = packed_data_collator(rows)

    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}

    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = float(model(**batch).loss)
    if was_training:
        model.train()

    print("\nInitial-loss sanity probe:")
    print(f"  vocab size            : {vocab_size:,}")
    print(f"  uniform-random loss   : {random_loss:.3f}  (ppl {vocab_size:,})")
    print(f"  observed initial loss : {loss:.3f}  (ppl {math.exp(min(loss, 20)):,.1f})")

    if not math.isfinite(loss):
        raise RuntimeError(
            "Initial loss is not finite. The forward pass is broken -- do not train. "
            "If precision is fp16, this is most likely Gemma 2 logit soft-capping "
            "overflow; use bf16 (L4/A100) instead."
        )

    if loss >= random_loss:
        raise RuntimeError(
            f"Initial loss {loss:.3f} is at or above the uniform-random baseline "
            f"{random_loss:.3f}. A correctly loaded pretrained model cannot score this "
            f"badly on in-domain text, so the model or the data is broken. Check, in "
            f"order: (1) the packing report says MODEL: {MODEL_NAME} -- packing with the "
            f"wrong tokenizer produces exactly this; (2) the loader -- see --loader, this "
            f"repo has seen a silently corrupt Unsloth forward for Gemma 2. "
            f"Aborting instead of training an unusable adapter."
        )

    if loss > 0.75 * random_loss:
        print(
            f"  WARNING: {loss:.3f} is high (>75% of random). Training may still work, "
            f"but verify the packing tokenizer before committing a long run."
        )
    else:
        print("  OK -- well below random, forward pass looks healthy.")

    return {"initial_loss": round(loss, 6), "random_loss": round(random_loss, 6)}


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Full Gemma 2-2B QLoRA continued pretraining."
    )

    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--train_file", default=TRAIN_FILE)
    parser.add_argument("--val_file", default=VAL_FILE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)

    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)

    parser.add_argument(
        "--loader",
        choices=["transformers", "unsloth"],
        default="transformers",
        help="transformers (default): plain QLoRA, the path validated for Gemma 2 in "
             "this repo. unsloth: the original bundle path -- faster, but see the "
             "module docstring before trusting it on Gemma 2.",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16"],
        default="auto",
        help="auto picks bf16 when the GPU supports it, else fp16. Gemma 2's logit "
             "soft-capping makes fp16 risky over a long run -- prefer an L4/A100.",
    )
    parser.add_argument(
        "--skip_sanity_probe",
        action="store_true",
        help="Skip the initial-loss check. Not recommended; it costs seconds and "
             "catches a silently broken forward pass.",
    )

    # How long to train. Epochs is the normal knob; max_steps > 0 overrides it.
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)

    # Effective batch = batch_size * grad_accum * num_gpus.
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)

    # Optimization.
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", default="cosine")

    # LoRA (defaults match the validated 12B config).
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # Eval / checkpoint cadence, in optimizer steps.
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=200,
        help="Cap on val rows used for in-training eval, for speed. 0 uses all. "
             "The thesis perplexity number comes from the full held-out set via "
             "next_token_eval_gemma.py, not from this.",
    )

    # Housekeeping.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--report_to", default="none")

    # Resume: "auto" resumes from the latest checkpoint if one exists,
    # "none" starts fresh, or pass an explicit checkpoint path.
    parser.add_argument("--resume", default="auto")

    # Validate the entire pipeline on a tiny slice before the real run.
    parser.add_argument("--dry_run_samples", type=int, default=0)

    # S3 checkpoint backup. Unused on Colab -- always pass --skip_s3_upload.
    parser.add_argument("--s3_bucket", default=None)
    parser.add_argument("--s3_prefix", default="gemma2_2b_cpt_v1")
    parser.add_argument("--s3_region", default=None)
    parser.add_argument("--skip_s3_upload", action="store_true")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# S3 checkpoint backup (unchanged; inert when --skip_s3_upload is passed)
# -----------------------------------------------------------------------------
def upload_directory_to_s3(local_dir, bucket, s3_prefix, s3_client):
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Cannot upload missing directory: {local_dir}")

    uploaded_files = 0
    for local_file in sorted(local_dir.rglob("*")):
        if not local_file.is_file():
            continue

        relative_path = local_file.relative_to(local_dir).as_posix()
        s3_key = f"{s3_prefix.rstrip('/')}/{relative_path}"

        s3_client.upload_file(str(local_file), bucket, s3_key)
        uploaded_files += 1

    if uploaded_files == 0:
        raise RuntimeError(f"No files found to upload in: {local_dir}")

    return uploaded_files


class S3CheckpointCallback(TrainerCallback):
    """Upload each checkpoint immediately after Trainer saves it locally."""

    def __init__(self, bucket, prefix, region=None, enabled=True):
        if enabled and boto3 is None:
            raise ImportError(
                "boto3 is not installed but S3 upload was requested. "
                "Pass --skip_s3_upload (the expected mode on Colab)."
            )
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.enabled = enabled
        self.s3_client = boto3.client("s3", region_name=region) if enabled else None

    def on_save(self, args, state, control, **kwargs):
        if not self.enabled:
            return control

        checkpoint_name = f"checkpoint-{state.global_step}"
        checkpoint_dir = Path(args.output_dir) / checkpoint_name

        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Trainer reported a save, but checkpoint is missing: {checkpoint_dir}"
            )

        checkpoint_prefix = f"{self.prefix}/checkpoints/{checkpoint_name}"
        print(f"\nUploading {checkpoint_dir} to s3://{self.bucket}/{checkpoint_prefix}/")

        uploaded_files = upload_directory_to_s3(
            local_dir=checkpoint_dir,
            bucket=self.bucket,
            s3_prefix=checkpoint_prefix,
            s3_client=self.s3_client,
        )
        print(f"S3 upload complete for {checkpoint_name}: {uploaded_files} files.")
        return control


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    if not args.skip_s3_upload and not args.s3_bucket:
        raise ValueError("--s3_bucket is required unless --skip_s3_upload is used.")

    output_path = Path(args.output_dir)

    print("=" * 80)
    print("GEMMA 2-2B - FULL CPT (QLoRA)")
    print("=" * 80)
    print("Model:", args.model_name)
    print("Loader:", args.loader)
    print("Train file:", args.train_file)
    print("Val file:", args.val_file)
    print("Output dir:", args.output_dir)
    print("Max seq length:", args.max_seq_length)
    print("Checkpoint cadence:", args.save_steps, "optimizer steps")
    print("Local checkpoint limit:", args.save_total_limit)
    print("S3 upload enabled:", not args.skip_s3_upload)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    gpu_name = torch.cuda.get_device_name(0)
    print("GPU:", gpu_name)

    # --- precision -----------------------------------------------------------
    bf16_ok = torch.cuda.is_bf16_supported()
    if args.precision == "auto":
        use_bf16 = bf16_ok
    else:
        use_bf16 = args.precision == "bf16"

    if use_bf16 and not bf16_ok:
        raise RuntimeError(
            f"bf16 requested but {gpu_name} does not support it. "
            "Restart the Colab runtime until you get an L4 or A100 -- that is the "
            "first move, not a config change. If a T4 is unavoidable, pass "
            "--precision fp16 and read Part 7.4 of Week2_CPT_Plan.md first."
        )
    if not use_bf16:
        print(
            "\nWARNING: training in fp16. Gemma 2 uses logit soft-capping "
            "(attention 50.0, final 30.0) and fp16 overflow is a real failure mode "
            "on this family. Watch the first 200 steps for a loss spike or NaN."
        )

    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print("Precision:", "bf16" if use_bf16 else "fp16")

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}. "
            "Packed files are not committed to git -- regenerate them with "
            "02_pack_dataset_4096.py before launching."
        )
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    # -------------------------------------------------------------------------
    # Model + LoRA
    # -------------------------------------------------------------------------
    print(f"\nLoading model via '{args.loader}'...")
    if args.loader == "unsloth":
        model, tokenizer = load_with_unsloth(args)
    else:
        model, tokenizer = load_with_transformers(args, compute_dtype)

    print("Model loaded. Tokenizer class:", type(tokenizer).__name__)

    # Confirm LoRA attached. ~0.5-1.5% trainable is expected; 0% or ~100% means
    # it did not land.
    model.print_trainable_parameters()

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    print("\nLoading packed dataset...")
    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    if args.dry_run_samples > 0:
        train_n = min(args.dry_run_samples, len(train_dataset))
        val_n = min(max(8, args.dry_run_samples // 8), len(val_dataset))
        print(f"DRY RUN: using {train_n} train / {val_n} val rows (pipeline validation only).")
        train_dataset = train_dataset.select(range(train_n))
        val_dataset = val_dataset.select(range(val_n))
    elif args.max_eval_samples > 0 and len(val_dataset) > args.max_eval_samples:
        print(
            f"Capping in-training eval at {args.max_eval_samples} of "
            f"{len(val_dataset)} val rows (see --max_eval_samples)."
        )
        val_dataset = val_dataset.select(range(args.max_eval_samples))

    print("Train rows:", len(train_dataset))
    print("Val rows:", len(val_dataset))

    validate_example(train_dataset[0], "train", 0, args.max_seq_length)
    validate_example(val_dataset[0], "validation", 0, args.max_seq_length)

    train_dataset = strip_non_training_columns(train_dataset, "train")
    val_dataset = strip_non_training_columns(val_dataset, "validation")

    # -------------------------------------------------------------------------
    # Sanity probe -- before anything expensive
    # -------------------------------------------------------------------------
    probe = None
    if not args.skip_sanity_probe:
        probe = initial_loss_probe(model, train_dataset)

    # -------------------------------------------------------------------------
    # Step / time accounting (for the log; not used to cap training)
    # -------------------------------------------------------------------------
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)

    if args.max_steps and args.max_steps > 0:
        planned_steps = args.max_steps
    else:
        planned_steps = math.ceil(steps_per_epoch * args.num_epochs)

    print("\nRun plan:")
    print("Effective batch (rows/optimizer step):", effective_batch)
    print("Optimizer steps per epoch:", steps_per_epoch)
    print("Planned optimizer steps:", planned_steps)
    print("Tokens per epoch (approx):", f"{len(train_dataset) * args.max_seq_length:,}")

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,

        bf16=use_bf16,
        fp16=not use_bf16,

        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,

        eval_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        # Keep the checkpoint with the lowest eval_loss at the end of training.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        optim="adamw_8bit",

        report_to=args.report_to,

        remove_unused_columns=False,
        prediction_loss_only=True,

        gradient_checkpointing=False,  # already enabled on the model itself
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
    )

    callbacks = []
    if not args.skip_s3_upload:
        callbacks.append(
            S3CheckpointCallback(
                bucket=args.s3_bucket,
                prefix=args.s3_prefix,
                region=args.s3_region,
                enabled=True,
            )
        )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=packed_data_collator,
        callbacks=callbacks,
    )
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        # Older transformers used tokenizer= instead of processing_class=.
        print("processing_class= not accepted; falling back to tokenizer=.")
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    # -------------------------------------------------------------------------
    # Resume handling
    # -------------------------------------------------------------------------
    resume_from = None
    if args.resume == "auto":
        if output_path.exists():
            last_checkpoint = get_last_checkpoint(str(output_path))
            if last_checkpoint is not None:
                resume_from = last_checkpoint
                print(f"\nResuming from checkpoint: {resume_from}")
            else:
                print("\nNo existing checkpoint found; starting fresh.")
    elif args.resume in ("none", "no", "false", ""):
        print("\nResume disabled; starting fresh.")
    else:
        resume_from = args.resume
        print(f"\nResuming from explicit checkpoint: {resume_from}")

    output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    print("\nStarting full CPT training...")
    torch.cuda.reset_peak_memory_stats()

    training_start = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    training_end = time.perf_counter()

    total_seconds = training_end - training_start
    completed_steps = int(train_result.global_step) or 1
    seconds_per_step = total_seconds / completed_steps

    print("\nTraining result:")
    print(train_result)

    # -------------------------------------------------------------------------
    # Final evaluation + save
    # -------------------------------------------------------------------------
    print("\nRunning final evaluation...")
    final_eval = trainer.evaluate()
    print("Final eval:", final_eval)

    final_loss = final_eval.get("eval_loss")
    if final_loss is None:
        raise ValueError("Final eval loss is missing.")
    if not math.isfinite(float(final_loss)):
        raise ValueError(f"Final eval loss is not finite: {final_loss}")

    final_dir = output_path / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving final adapter to:", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    adapter_config_path = final_dir / "adapter_config.json"
    adapter_weights_path = final_dir / "adapter_model.safetensors"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing saved adapter config: {adapter_config_path}")
    if not adapter_weights_path.exists():
        raise FileNotFoundError(f"Missing saved adapter weights: {adapter_weights_path}")

    if not args.skip_s3_upload:
        final_s3_prefix = f"{args.s3_prefix.strip('/')}/final_adapter"
        print(f"\nUploading final adapter to s3://{args.s3_bucket}/{final_s3_prefix}/")
        final_s3_client = boto3.client("s3", region_name=args.s3_region)
        uploaded_files = upload_directory_to_s3(
            local_dir=final_dir,
            bucket=args.s3_bucket,
            s3_prefix=final_s3_prefix,
            s3_client=final_s3_client,
        )
        print(f"Final adapter S3 upload complete: {uploaded_files} files.")

    peak_gpu_gb = torch.cuda.max_memory_allocated() / 1024**3

    # Write a small run summary next to the adapter.
    summary = {
        "model_name": args.model_name,
        "loader": args.loader,
        "precision": "bf16" if use_bf16 else "fp16",
        "gpu": gpu_name,
        "output_dir": args.output_dir,
        "max_seq_length": args.max_seq_length,
        "num_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "effective_batch": effective_batch,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": TARGET_MODULES,
        "completed_steps": completed_steps,
        "final_eval_loss": round(float(final_loss), 6),
        "final_eval_perplexity": round(math.exp(min(float(final_loss), 20)), 4),
        "total_training_seconds": round(total_seconds, 2),
        "seconds_per_optimizer_step": round(seconds_per_step, 2),
        "peak_gpu_memory_gb": round(peak_gpu_gb, 2),
        "dry_run_samples": args.dry_run_samples,
    }
    if probe:
        summary.update(probe)

    # Versions matter: this repo has been bitten by a silently broken stack before.
    try:
        import transformers, peft
        summary["versions"] = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        }
        if args.loader == "unsloth":
            import unsloth
            summary["versions"]["unsloth"] = unsloth.__version__
    except Exception as e:  # never fail the run over bookkeeping
        summary["versions"] = {"error": str(e)}

    with (final_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSUCCESS.")
    print("Final adapter:", final_dir)
    print("Final eval loss:", round(float(final_loss), 6))
    print("Final eval perplexity:", summary["final_eval_perplexity"])
    print("Completed optimizer steps:", completed_steps)
    print("Total training seconds:", round(total_seconds, 2))
    print("Seconds per optimizer step:", round(seconds_per_step, 2))
    print("Peak allocated GPU memory:", round(peak_gpu_gb, 2), "GB")
    print(
        "Estimated seconds per epoch:",
        round(steps_per_epoch * seconds_per_step, 2),
        f"(~{round(steps_per_epoch * seconds_per_step / 3600, 2)} hours)",
    )


if __name__ == "__main__":
    main()
