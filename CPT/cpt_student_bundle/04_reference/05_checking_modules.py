"""
Inspect LoRA module placement for Gemma 4 CPT.

Purpose
-------
1. Load google/gemma-4-12B using the existing working Unsloth setup.
2. Attach LoRA using the current CPT configuration.
3. Confirm the number of trainable parameters.
4. List every module that received LoRA.
5. Detect possible vision/audio adapters.
6. Save detailed reports.
7. Exit without loading datasets or starting training.

This script does not modify the model files or start CPT.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Set cache locations before importing libraries that use Hugging Face.
NVME_CACHE = Path("/opt/dlami/nvme/huggingface")

os.environ.setdefault("HF_HOME", str(NVME_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(NVME_CACHE / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(NVME_CACHE / "hub"))

import unsloth
import torch
import transformers
from unsloth import FastLanguageModel


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "google/gemma-4-12B"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0
RANDOM_STATE = 42

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

REPORT_DIRECTORY = Path("reports/lora_module_inspection")
TEXT_REPORT_PATH = REPORT_DIRECTORY / "lora_module_report.txt"
JSON_REPORT_PATH = REPORT_DIRECTORY / "lora_module_report.json"
TRAINABLE_PARAMETERS_PATH = (
    REPORT_DIRECTORY / "trainable_parameter_names.txt"
)


# ---------------------------------------------------------------------
# Classification keywords
# ---------------------------------------------------------------------

VISION_KEYWORDS = (
    "vision",
    "visual",
    "image",
    "images",
    "siglip",
    "clip",
    "vit",
    "pixel",
)

AUDIO_KEYWORDS = (
    "audio",
    "speech",
    "sound",
    "acoustic",
    "whisper",
)

LANGUAGE_KEYWORDS = (
    "language",
    "language_model",
    "text",
    "text_model",
    "decoder",
    "model.layers",
    "transformer",
    "self_attn",
    "mlp",
)

ATTENTION_MODULE_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
}

MLP_MODULE_NAMES = {
    "gate_proj",
    "up_proj",
    "down_proj",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def ensure_environment() -> None:
    """Check basic runtime requirements before loading the model."""

    print("=" * 80)
    print("ENVIRONMENT")
    print("=" * 80)

    print(f"Python:       {sys.version.split()[0]}")
    print(f"Platform:     {platform.platform()}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"CUDA usable:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this script on the EC2 GPU instance."
        )

    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(
        "GPU memory:   "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

    if not NVME_CACHE.exists():
        print(
            f"WARNING: NVMe cache path does not exist: {NVME_CACHE}"
        )

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def classify_module_name(module_name: str) -> str:
    """
    Classify a LoRA module using its complete path.

    Classification is only a diagnostic aid. The complete names are
    always saved for manual inspection.
    """

    lower_name = module_name.lower()

    if any(keyword in lower_name for keyword in VISION_KEYWORDS):
        return "vision"

    if any(keyword in lower_name for keyword in AUDIO_KEYWORDS):
        return "audio"

    if any(keyword in lower_name for keyword in LANGUAGE_KEYWORDS):
        return "language"

    return "unclassified"


def get_projection_type(module_name: str) -> str:
    """Determine which projection type a module represents."""

    final_component = module_name.split(".")[-1]

    if final_component in ATTENTION_MODULE_NAMES:
        return "attention"

    if final_component in MLP_MODULE_NAMES:
        return "mlp"

    return "other"


def get_lora_module_names(model: torch.nn.Module) -> list[str]:
    """
    Return modules that contain LoRA A or B adapters.

    PEFT normally places lora_A and lora_B containers on the wrapped
    linear module.
    """

    lora_module_names: list[str] = []

    for module_name, module in model.named_modules():
        has_lora_a = hasattr(module, "lora_A")
        has_lora_b = hasattr(module, "lora_B")

        if has_lora_a or has_lora_b:
            lora_module_names.append(module_name)

    # Remove duplicate names while preserving order.
    return list(dict.fromkeys(lora_module_names))


def get_trainable_parameter_information(
    model: torch.nn.Module,
) -> tuple[list[dict[str, Any]], int, int]:
    """Collect all trainable parameter names and sizes."""

    trainable_parameters: list[dict[str, Any]] = []
    trainable_count = 0
    total_count = 0

    for parameter_name, parameter in model.named_parameters():
        parameter_count = parameter.numel()
        total_count += parameter_count

        if parameter.requires_grad:
            trainable_count += parameter_count

            trainable_parameters.append(
                {
                    "name": parameter_name,
                    "shape": list(parameter.shape),
                    "num_parameters": parameter_count,
                    "dtype": str(parameter.dtype),
                }
            )

    return trainable_parameters, trainable_count, total_count


def get_top_level_prefix(module_name: str, depth: int = 5) -> str:
    """
    Return a shortened path for grouping modules.

    Example:
    model.language_model.layers.0.self_attn.q_proj
    becomes:
    model.language_model.layers.0.self_attn
    """

    parts = module_name.split(".")
    return ".".join(parts[:depth])


def inspect_lora_modules(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Inspect and summarize the LoRA attachment locations."""

    lora_module_names = get_lora_module_names(model)

    if not lora_module_names:
        raise RuntimeError(
            "No LoRA modules were found. "
            "The adapter may not have attached correctly."
        )

    classified_modules: dict[str, list[str]] = defaultdict(list)
    projection_counts: Counter[str] = Counter()
    exact_projection_counts: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()

    for module_name in lora_module_names:
        classification = classify_module_name(module_name)
        projection_type = get_projection_type(module_name)
        final_component = module_name.split(".")[-1]

        classified_modules[classification].append(module_name)
        projection_counts[projection_type] += 1
        exact_projection_counts[final_component] += 1
        prefix_counts[get_top_level_prefix(module_name)] += 1

    (
        trainable_parameters,
        trainable_count,
        total_count,
    ) = get_trainable_parameter_information(model)

    trainable_percentage = (
        100.0 * trainable_count / total_count
        if total_count
        else 0.0
    )

    suspicious_trainable_parameters = []

    for parameter in trainable_parameters:
        classification = classify_module_name(parameter["name"])

        if classification in {"vision", "audio"}:
            suspicious_trainable_parameters.append(
                {
                    **parameter,
                    "classification": classification,
                }
            )

    report: dict[str, Any] = {
        "configuration": {
            "model_name": MODEL_NAME,
            "max_seq_length": MAX_SEQ_LENGTH,
            "load_in_4bit": LOAD_IN_4BIT,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
            "random_state": RANDOM_STATE,
        },
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "summary": {
            "total_lora_wrapped_modules": len(lora_module_names),
            "language_like_modules": len(
                classified_modules["language"]
            ),
            "vision_like_modules": len(
                classified_modules["vision"]
            ),
            "audio_like_modules": len(
                classified_modules["audio"]
            ),
            "unclassified_modules": len(
                classified_modules["unclassified"]
            ),
            "trainable_parameters": trainable_count,
            "total_parameters": total_count,
            "trainable_percentage": trainable_percentage,
            "suspicious_trainable_parameters": len(
                suspicious_trainable_parameters
            ),
        },
        "projection_counts": dict(
            sorted(exact_projection_counts.items())
        ),
        "projection_group_counts": dict(
            sorted(projection_counts.items())
        ),
        "path_prefix_counts": dict(
            prefix_counts.most_common()
        ),
        "modules": {
            "all": lora_module_names,
            "language": classified_modules["language"],
            "vision": classified_modules["vision"],
            "audio": classified_modules["audio"],
            "unclassified": classified_modules["unclassified"],
        },
        "trainable_parameter_details": trainable_parameters,
        "suspicious_trainable_parameter_details": (
            suspicious_trainable_parameters
        ),
    }

    return report


def save_json_report(report: dict[str, Any]) -> None:
    """Save the complete structured report."""

    with JSON_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            report,
            output_file,
            ensure_ascii=False,
            indent=2,
        )


def save_trainable_parameter_names(
    report: dict[str, Any],
) -> None:
    """Save all trainable parameter names and shapes."""

    with TRAINABLE_PARAMETERS_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for parameter in report["trainable_parameter_details"]:
            output_file.write(
                f'{parameter["name"]}\n'
                f'  shape={parameter["shape"]}\n'
                f'  count={parameter["num_parameters"]:,}\n'
                f'  dtype={parameter["dtype"]}\n\n'
            )


def save_text_report(report: dict[str, Any]) -> None:
    """Save a human-readable report."""

    summary = report["summary"]
    modules = report["modules"]

    with TEXT_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write("LORA MODULE INSPECTION REPORT\n")
        output_file.write("=" * 80 + "\n\n")

        output_file.write("CONFIGURATION\n")
        output_file.write("-" * 80 + "\n")

        for key, value in report["configuration"].items():
            output_file.write(f"{key}: {value}\n")

        output_file.write("\nSUMMARY\n")
        output_file.write("-" * 80 + "\n")

        for key, value in summary.items():
            if key == "trainable_percentage":
                output_file.write(f"{key}: {value:.6f}%\n")
            elif isinstance(value, int):
                output_file.write(f"{key}: {value:,}\n")
            else:
                output_file.write(f"{key}: {value}\n")

        output_file.write("\nPROJECTION COUNTS\n")
        output_file.write("-" * 80 + "\n")

        for projection, count in report["projection_counts"].items():
            output_file.write(f"{projection}: {count}\n")

        output_file.write("\nPATH PREFIX COUNTS\n")
        output_file.write("-" * 80 + "\n")

        for prefix, count in report["path_prefix_counts"].items():
            output_file.write(f"{prefix}: {count}\n")

        for category in (
            "language",
            "vision",
            "audio",
            "unclassified",
        ):
            output_file.write(
                f"\n{category.upper()} LORA MODULES "
                f"({len(modules[category])})\n"
            )
            output_file.write("-" * 80 + "\n")

            if modules[category]:
                for module_name in modules[category]:
                    output_file.write(module_name + "\n")
            else:
                output_file.write("None found.\n")

        output_file.write("\nALL LORA MODULES\n")
        output_file.write("-" * 80 + "\n")

        for module_name in modules["all"]:
            output_file.write(module_name + "\n")

        output_file.write(
            "\nSUSPICIOUS VISION/AUDIO TRAINABLE PARAMETERS\n"
        )
        output_file.write("-" * 80 + "\n")

        suspicious_parameters = report[
            "suspicious_trainable_parameter_details"
        ]

        if suspicious_parameters:
            for parameter in suspicious_parameters:
                output_file.write(
                    f'{parameter["classification"]}: '
                    f'{parameter["name"]} '
                    f'shape={parameter["shape"]} '
                    f'count={parameter["num_parameters"]:,}\n'
                )
        else:
            output_file.write(
                "No obvious vision/audio trainable parameters found.\n"
            )


def print_report_summary(report: dict[str, Any]) -> None:
    """Print the most important findings."""

    summary = report["summary"]

    print("\n" + "=" * 80)
    print("LORA INSPECTION SUMMARY")
    print("=" * 80)

    print(
        "Total LoRA-wrapped modules: "
        f'{summary["total_lora_wrapped_modules"]:,}'
    )
    print(
        "Language-like modules:      "
        f'{summary["language_like_modules"]:,}'
    )
    print(
        "Vision-like modules:        "
        f'{summary["vision_like_modules"]:,}'
    )
    print(
        "Audio-like modules:         "
        f'{summary["audio_like_modules"]:,}'
    )
    print(
        "Unclassified modules:       "
        f'{summary["unclassified_modules"]:,}'
    )

    print(
        "\nTrainable parameters:        "
        f'{summary["trainable_parameters"]:,}'
    )
    print(
        "Total parameters:            "
        f'{summary["total_parameters"]:,}'
    )
    print(
        "Trainable percentage:        "
        f'{summary["trainable_percentage"]:.6f}%'
    )

    print("\nProjection counts:")

    for projection, count in report["projection_counts"].items():
        print(f"  {projection:<12} {count:,}")

    vision_modules = report["modules"]["vision"]
    audio_modules = report["modules"]["audio"]
    unclassified_modules = report["modules"]["unclassified"]

    if vision_modules:
        print("\nWARNING: possible vision LoRA modules found:")

        for module_name in vision_modules[:20]:
            print(f"  {module_name}")

        if len(vision_modules) > 20:
            print(
                f"  ... plus {len(vision_modules) - 20} more"
            )

    if audio_modules:
        print("\nWARNING: possible audio LoRA modules found:")

        for module_name in audio_modules[:20]:
            print(f"  {module_name}")

        if len(audio_modules) > 20:
            print(
                f"  ... plus {len(audio_modules) - 20} more"
            )

    if unclassified_modules:
        print(
            "\nNOTE: Some modules were not classified automatically."
        )
        print("Review these names manually:")

        for module_name in unclassified_modules[:20]:
            print(f"  {module_name}")

        if len(unclassified_modules) > 20:
            print(
                f"  ... plus {len(unclassified_modules) - 20} more"
            )

    if not vision_modules and not audio_modules:
        print(
            "\nRESULT: No obvious vision or audio LoRA modules "
            "were detected."
        )

    print("\nReports saved to:")
    print(f"  {TEXT_REPORT_PATH}")
    print(f"  {JSON_REPORT_PATH}")
    print(f"  {TRAINABLE_PARAMETERS_PATH}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Load the model, attach LoRA, inspect, save reports, and exit."""

    ensure_environment()

    print("\n" + "=" * 80)
    print("LOADING BASE MODEL")
    print("=" * 80)
    print(f"Model:          {MODEL_NAME}")
    print(f"Sequence length:{MAX_SEQ_LENGTH}")
    print(f"4-bit loading:  {LOAD_IN_4BIT}")

    model, processor = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )

    print("\nBase model loaded successfully.")
    print(f"Model class:     {type(model).__name__}")
    print(f"Processor class: {type(processor).__name__}")

    tokenizer = getattr(processor, "tokenizer", processor)

    print(
        "EOS token:       "
        f"{getattr(tokenizer, 'eos_token', None)!r} / "
        f"{getattr(tokenizer, 'eos_token_id', None)!r}"
    )
    print(
        "PAD token:       "
        f"{getattr(tokenizer, 'pad_token', None)!r} / "
        f"{getattr(tokenizer, 'pad_token_id', None)!r}"
    )

    print("\n" + "=" * 80)
    print("ATTACHING LORA")
    print("=" * 80)
    print(f"Rank:           {LORA_RANK}")
    print(f"Alpha:          {LORA_ALPHA}")
    print(f"Dropout:        {LORA_DROPOUT}")
    print(f"Target modules: {TARGET_MODULES}")

    # Keep the exact working LoRA setup for the first inspection.
    # Do not add tower-freezing options until we know whether they
    # are actually needed.
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_STATE,
    )

    print("\nPEFT trainable-parameter summary:")
    model.print_trainable_parameters()

    report = inspect_lora_modules(model)

    save_text_report(report)
    save_json_report(report)
    save_trainable_parameter_names(report)
    print_report_summary(report)

    print("\nInspection finished. No training was started.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInspection cancelled by user.")
        raise SystemExit(130)
    except Exception as error:
        print("\n" + "=" * 80)
        print("INSPECTION FAILED")
        print("=" * 80)
        print(f"{type(error).__name__}: {error}")
        raise