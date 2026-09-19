import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer


MODEL_NAME = "google/gemma-4-12B"

TRAIN_FILE = (
    "data_cpt/final_cpt_v1/packed/"
    "train_cleanheaders_packed_4096.jsonl"
)
VAL_FILE = (
    "data_cpt/final_cpt_v1/packed/"
    "val_cleanheaders_packed_4096.jsonl"
)

OUTPUT_DIR = "adapters/test_unsloth_gemma4_12b_packed_2steps"

MAX_SEQ_LENGTH = 4096
MAX_STEPS = 2

BATCH_SIZE = 1
GRAD_ACCUM = 8

LEARNING_RATE = 2e-4


def packed_data_collator(features):
    """
    Collator for fixed-length packed CPT examples.

    The JSONL records already contain:
      - input_ids
      - attention_mask

    All sequences are expected to have length 4096, so no tokenizer
    padding is needed. Labels are copied from input_ids, and any
    padding positions are masked with -100.
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


def validate_example(example, split_name, index):
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

    if input_length != MAX_SEQ_LENGTH:
        raise ValueError(
            f"{split_name} row {index} has length {input_length}, "
            f"expected {MAX_SEQ_LENGTH}"
        )


def main():
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"UNSLOTH QUICK TEST - {MAX_STEPS} OPTIMIZER STEPS")
    print("=" * 80)

    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Initial GPU memory allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB",
    )

    print("\nLoading model...")
    model, processor = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print("Model loaded successfully.")
    print("Processor class:", type(processor).__name__)

    print("\nAdding LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()
    print("LoRA attached successfully.")

    print("\nLoading packed dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VAL_FILE,
        },
    )

    train_count = min(20, len(dataset["train"]))
    val_count = min(8, len(dataset["validation"]))

    train_dataset = dataset["train"].select(range(train_count))
    val_dataset = dataset["validation"].select(range(val_count))

    print("Train rows used:", len(train_dataset))
    print("Val rows used:", len(val_dataset))

    validate_example(train_dataset[0], "train", 0)
    validate_example(val_dataset[0], "validation", 0)

    print(
        "First train input_ids length:",
        len(train_dataset[0]["input_ids"]),
    )
    print(
        "First validation input_ids length:",
        len(val_dataset[0]["input_ids"]),
    )

    keep_columns = {"input_ids", "attention_mask"}

    train_columns_to_remove = [
        column
        for column in train_dataset.column_names
        if column not in keep_columns
    ]

    val_columns_to_remove = [
        column
        for column in val_dataset.column_names
        if column not in keep_columns
    ]

    if train_columns_to_remove:
        print(
            "Removing non-training train columns:",
            train_columns_to_remove,
        )
        train_dataset = train_dataset.remove_columns(
            train_columns_to_remove
        )

    if val_columns_to_remove:
        print(
            "Removing non-training validation columns:",
            val_columns_to_remove,
        )
        val_dataset = val_dataset.remove_columns(
            val_columns_to_remove
        )

    print("Trainer train columns:", train_dataset.column_names)
    print("Trainer validation columns:", val_dataset.column_names)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        max_steps=MAX_STEPS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,

        learning_rate=LEARNING_RATE,
        warmup_steps=1,

        bf16=True,
        fp16=False,

        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,

        # MAX_STEPS is currently 2, so evaluation during training
        # would not run with eval_steps=5. We perform one final
        # evaluation explicitly after training instead.
        eval_strategy="no",

        # Avoid intermediate checkpoints during the smoke test.
        save_strategy="no",

        optim="adamw_8bit",

        report_to="none",

        remove_unused_columns=False,
        prediction_loss_only=True,

        dataloader_num_workers=0,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=packed_data_collator,
    )

    print(f"\nStarting {MAX_STEPS}-step training test...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    training_start = time.perf_counter()
    train_result = trainer.train()
    training_end = time.perf_counter()

    total_seconds = training_end - training_start
    seconds_per_step = total_seconds / MAX_STEPS

    print("\nTraining result:")
    print(train_result)

    print("\nRunning final evaluation...")
    evaluation_start = time.perf_counter()
    final_eval = trainer.evaluate()
    evaluation_end = time.perf_counter()

    print("Final eval:", final_eval)
    print(
        "Evaluation seconds:",
        round(evaluation_end - evaluation_start, 2),
    )

    final_loss = final_eval.get("eval_loss")

    if final_loss is None:
        raise ValueError("Final eval loss is missing.")

    if not math.isfinite(float(final_loss)):
        raise ValueError(
            f"Final eval loss is not finite: {final_loss}"
        )

    print("\nSaving adapter...")
    trainer.save_model(OUTPUT_DIR)

    # Gemma 4 returns a processor rather than a plain tokenizer.
    # save_pretrained is still the correct method.
    processor.save_pretrained(OUTPUT_DIR)

    adapter_config_path = output_path / "adapter_config.json"
    adapter_weights_path = output_path / "adapter_model.safetensors"

    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"Missing saved adapter config: {adapter_config_path}"
        )

    if not adapter_weights_path.exists():
        raise FileNotFoundError(
            f"Missing saved adapter weights: {adapter_weights_path}"
        )

    peak_gpu_gb = (
        torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else 0
    )

    print("\nSUCCESS.")
    print("Adapter saved to:", OUTPUT_DIR)
    print("Final eval loss:", round(float(final_loss), 6))
    print("Total training seconds:", round(total_seconds, 2))
    print(
        "Seconds per optimizer step:",
        round(seconds_per_step, 2),
    )
    print(
        "Minutes per optimizer step:",
        round(seconds_per_step / 60, 2),
    )
    print(
        "Peak allocated GPU memory:",
        round(peak_gpu_gb, 2),
        "GB",
    )

    full_optimizer_steps = 3894
    estimated_full_seconds = (
        full_optimizer_steps * seconds_per_step
    )

    print("\nEstimated full training time:")
    print("Optimizer steps:", full_optimizer_steps)
    print(
        "Estimated hours:",
        round(estimated_full_seconds / 3600, 2),
    )
    print(
        "Estimated days:",
        round(estimated_full_seconds / 86400, 2),
    )

    print(
        "\nNote: the first step may include compilation overhead. "
        "A longer 5-step run gives a better steady-state estimate."
    )


if __name__ == "__main__":
    main()