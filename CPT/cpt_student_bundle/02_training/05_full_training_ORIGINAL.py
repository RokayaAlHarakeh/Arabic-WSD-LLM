"""
Full continued-pretraining (CPT) run for Gemma 4 12B on the packed
Lebanese-legal dataset, using Unsloth + 4-bit QLoRA.

This is the production counterpart to 03_smoke_test_cpt.py. It reuses the
exact same, already-validated setup from the smoke test:

  - FastLanguageModel loader (returns a processor for Gemma 4)
  - custom packed_data_collator (no tokenizer padding; labels masked by
    attention_mask)
  - explicit removal of non-training columns (id, packed, max_seq_length)
  - LoRA config (r=16, 7 target modules, unsloth gradient checkpointing)
  - adamw_8bit optimizer, bf16

and adds what a full run needs:

  - training by epochs (not a fixed step cap)
  - cosine LR schedule with warmup, weight decay, grad clipping
  - periodic evaluation and checkpointing
  - automatic resume from the latest checkpoint
  - best-model tracking on eval_loss
  - a --dry_run_samples mode to validate the whole path end-to-end quickly
  - a trainable-parameters print so you can confirm LoRA landed on the
    text tower only

Run (all defaults, single GPU):

    python training_codes/05_full_cpt_training.py

Quick end-to-end validation on a tiny slice before the real run:

    python training_codes/05_full_cpt_training.py --dry_run_samples 64 --num_epochs 1

Resume is automatic: re-running the same command picks up the latest
checkpoint in the output directory.
"""
import os

os.environ.setdefault(
    "HF_HOME",
    "/opt/dlami/nvme/huggingface",
)
os.environ.setdefault(
    "HF_HUB_CACHE",
    "/opt/dlami/nvme/huggingface/hub",
)
os.environ.setdefault(
    "TRANSFORMERS_CACHE",
    "/opt/dlami/nvme/huggingface/hub",
)
import argparse
import json
import math
import time
from pathlib import Path

import boto3

import torch
from datasets import load_dataset

# Unsloth must be imported before transformers so its patches apply.
from unsloth import FastLanguageModel

from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
#TrainingArguments defines how training should run, Trainer performs the run, and get_last_checkpoint allows recovery after interruption.

# -----------------------------------------------------------------------------
# Defaults (override any of these on the command line)
# -----------------------------------------------------------------------------
MODEL_NAME = "google/gemma-4-12B"

TRAIN_FILE = (
    "data_cpt/final_cpt_v1/packed/"
    "train_cleanheaders_packed_4096.jsonl"
)
VAL_FILE = (
    "data_cpt/final_cpt_v1/packed/"
    "val_cleanheaders_packed_4096.jsonl"
)

OUTPUT_DIR = "adapters/gemma4_12b_cpt_full_v1"

MAX_SEQ_LENGTH = 4096


# -----------------------------------------------------------------------------
# Data collator and validation (identical behavior to the smoke test)
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
# Arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Full Gemma 4 12B QLoRA continued pretraining."
    )

    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--train_file", default=TRAIN_FILE)
    parser.add_argument("--val_file", default=VAL_FILE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)

    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)

    # How long to train. Epochs is the normal knob; max_steps > 0 overrides it.
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)

    # Effective batch = batch_size * grad_accum * num_gpus.
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=1)

    # Optimization.
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", default="cosine")

    # LoRA (defaults match the validated smoke-test config).
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # Eval / checkpoint cadence, in optimizer steps.
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=10)

    # Housekeeping.
    parser.add_argument("--seed", type=int, default=42)
    #here is no tokenization, text cleaning, image decoding, or heavy preprocessing during training. 
    # The collator only converts lists into tensors and creates labels. That work is very small, so extra workers are unlikely to make training noticeably faster.
    #Your bottleneck is the GPU computation on the 12B model, not loading one packed row from disk. 
    # While the GPU spends around a minute on an optimizer step, the CPU can prepare the next simple batch very quickly.
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--report_to", default="none")

    # Resume: "auto" resumes from the latest checkpoint if one exists,
    # "none" starts fresh, or pass an explicit checkpoint path.
    parser.add_argument("--resume", default="auto")

    # Validate the entire pipeline on a tiny slice before the real run.
    parser.add_argument("--dry_run_samples", type=int, default=0)

    # S3 checkpoint backup. Bucket is required for the production run.
    parser.add_argument(
        "--s3_bucket",
        default=None,
        help="S3 bucket name used to back up every saved checkpoint.",
    )
    parser.add_argument(
        "--s3_prefix",
        default="gemma4_12b_cpt_full_v1",
        help="Folder-like S3 prefix for checkpoints and the final adapter.",
    )
    parser.add_argument(
        "--s3_region",
        default=None,
        help="Optional AWS region. If omitted, boto3 uses the instance/AWS config.",
    )
    parser.add_argument(
        "--skip_s3_upload",
        action="store_true",
        help="Disable S3 uploads, useful only for a local dry run.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# S3 checkpoint backup
# -----------------------------------------------------------------------------
def upload_directory_to_s3(local_dir, bucket, s3_prefix, s3_client):
    """
    Upload every file under local_dir to s3://bucket/s3_prefix/.

    Files are uploaded with paths relative to local_dir, preserving the
    checkpoint directory structure.
    """
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Cannot upload missing directory: {local_dir}")

    uploaded_files = 0
    for local_file in sorted(local_dir.rglob("*")):
        if not local_file.is_file():
            continue

        relative_path = local_file.relative_to(local_dir).as_posix()
        s3_key = f"{s3_prefix.rstrip('/')}/{relative_path}"

        s3_client.upload_file(
            str(local_file),
            bucket,
            s3_key,
        )
        uploaded_files += 1

    if uploaded_files == 0:
        raise RuntimeError(f"No files found to upload in: {local_dir}")

    return uploaded_files


class S3CheckpointCallback(TrainerCallback):
    """
    Upload each checkpoint immediately after Trainer saves it locally.

    Local checkpoint rotation is still controlled by save_total_limit=5.
    Uploaded S3 checkpoints are not deleted by Trainer.
    """

    def __init__(self, bucket, prefix, region=None, enabled=True):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.enabled = enabled
        self.s3_client = (
            boto3.client("s3", region_name=region)
            if enabled
            else None
        )

    def on_save(self, args, state, control, **kwargs):
        if not self.enabled:
            return control

        checkpoint_name = f"checkpoint-{state.global_step}"
        checkpoint_dir = Path(args.output_dir) / checkpoint_name

        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Trainer reported a save, but checkpoint is missing: "
                f"{checkpoint_dir}"
            )

        checkpoint_prefix = f"{self.prefix}/checkpoints/{checkpoint_name}"

        print(
            f"\nUploading {checkpoint_dir} to "
            f"s3://{self.bucket}/{checkpoint_prefix}/"
        )

        uploaded_files = upload_directory_to_s3(
            local_dir=checkpoint_dir,
            bucket=self.bucket,
            s3_prefix=checkpoint_prefix,
            s3_client=self.s3_client,
        )

        print(
            f"S3 upload complete for {checkpoint_name}: "
            f"{uploaded_files} files."
        )
        return control


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    if not args.skip_s3_upload and not args.s3_bucket:
        raise ValueError(
            "--s3_bucket is required unless --skip_s3_upload is used."
        )

    output_path = Path(args.output_dir)

    print("=" * 80)
    print("GEMMA 4 12B - FULL CPT (QLoRA)")
    print("=" * 80)
    print("Model:", args.model_name)
    print("Train file:", args.train_file)
    print("Val file:", args.val_file)
    print("Output dir:", args.output_dir)
    print("Checkpoint cadence:", args.save_steps, "optimizer steps")
    print("Local checkpoint limit:", args.save_total_limit)
    print("S3 upload enabled:", not args.skip_s3_upload)
    if not args.skip_s3_upload:
        print("S3 destination:", f"s3://{args.s3_bucket}/{args.s3_prefix}/")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Initial GPU memory allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB",
    )

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}. "
            "The full packed train split is not committed to git; make sure "
            "it is present locally before launching a full run."
        )
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    # -------------------------------------------------------------------------
    # Model + LoRA
    # -------------------------------------------------------------------------
    print("\nLoading model...")
    model, processor = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print("Model loaded successfully.")
    print("Processor class:", type(processor).__name__)

    print("\nAdding LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    print("LoRA attached successfully.")

    # Sanity check: confirm LoRA adapters landed on the text tower only.
    # For Gemma 4 12B (48 text layers), expect a small trainable % and no
    # vision/audio modules in the trainable set.
    model.print_trainable_parameters()

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    print("\nLoading packed dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.val_file,
        },
    )

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    if args.dry_run_samples > 0:
        train_n = min(args.dry_run_samples, len(train_dataset))
        val_n = min(max(8, args.dry_run_samples // 8), len(val_dataset))
        print(
            f"DRY RUN: using {train_n} train / {val_n} val rows "
            "(pipeline validation only)."
        )
        train_dataset = train_dataset.select(range(train_n))
        val_dataset = val_dataset.select(range(val_n))

    print("Train rows:", len(train_dataset))
    print("Val rows:", len(val_dataset))

    validate_example(train_dataset[0], "train", 0, args.max_seq_length)
    validate_example(val_dataset[0], "validation", 0, args.max_seq_length)

    train_dataset = strip_non_training_columns(train_dataset, "train")
    val_dataset = strip_non_training_columns(val_dataset, "validation")

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
    print(
        "Tokens per epoch (approx):",
        f"{len(train_dataset) * args.max_seq_length:,}",
    )

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

        bf16=True,
        fp16=False,

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=packed_data_collator,
        callbacks=callbacks,
    )

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
    processor.save_pretrained(str(final_dir))

    adapter_config_path = final_dir / "adapter_config.json"
    adapter_weights_path = final_dir / "adapter_model.safetensors"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing saved adapter config: {adapter_config_path}")
    if not adapter_weights_path.exists():
        raise FileNotFoundError(f"Missing saved adapter weights: {adapter_weights_path}")

    if not args.skip_s3_upload:
        final_s3_prefix = f"{args.s3_prefix.strip('/')}/final_adapter"
        print(
            f"\nUploading final adapter to "
            f"s3://{args.s3_bucket}/{final_s3_prefix}/"
        )
        final_s3_client = boto3.client(
            "s3",
            region_name=args.s3_region,
        )
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
        "output_dir": args.output_dir,
        "num_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "effective_batch": effective_batch,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "completed_steps": completed_steps,
        "final_eval_loss": round(float(final_loss), 6),
        "total_training_seconds": round(total_seconds, 2),
        "seconds_per_optimizer_step": round(seconds_per_step, 2),
        "peak_gpu_memory_gb": round(peak_gpu_gb, 2),
        "dry_run_samples": args.dry_run_samples,
    }
    with (final_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSUCCESS.")
    print("Final adapter:", final_dir)
    print("Final eval loss:", round(float(final_loss), 6))
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