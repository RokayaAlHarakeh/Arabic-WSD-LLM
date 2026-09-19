#!/usr/bin/env python3
"""
Minimal Sonnet next-word Top-1 evaluation with Amazon Bedrock.

The output is one JSON file containing:
- top1_accuracy
- prefix
- target_word
- predicted_word
- correct

The original cleaned held-out JSONL is required because the frozen Gemma
target may be only part of a complete word.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from botocore.config import Config
from dotenv import load_dotenv
from langchain_aws import ChatBedrock


MODEL_ID = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"

ARABIC_DIACRITICS = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples_file", required=True)
    parser.add_argument("--source_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", "eu-west-1"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means all examples; use 10 for a smoke test.",
    )
    parser.add_argument("--sleep_seconds", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}, line {line_number}: expected JSON object"
                )
            yield line_number, value


def is_word_character(character: str) -> bool:
    category = unicodedata.category(character)
    return category.startswith(("L", "M", "N")) or character == "_"


def extract_next_unit(text: str, position: int) -> str:
    """Return one complete word/number or one punctuation symbol."""
    while position < len(text) and text[position].isspace():
        position += 1

    if position >= len(text):
        raise ValueError("No next unit exists after this position")

    start = position

    if is_word_character(text[position]):
        position += 1
        while (
            position < len(text)
            and is_word_character(text[position])
        ):
            position += 1
    else:
        position += 1

    return text[start:position]


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u0640", "")
    value = ARABIC_DIACRITICS.sub("", value)
    return value.strip()


def first_unit(value: str) -> str:
    value = value.strip().strip("`\"'“”‘’")
    return extract_next_unit(value, 0) if value else ""


def get_response_text(response: Any) -> str:
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts)

    return str(content)


def create_model(
    model_id: str,
    region: str,
) -> ChatBedrock:
    config = Config(
        connect_timeout=10,
        read_timeout=300,
        retries={"max_attempts": 3, "mode": "adaptive"},
    )

    return ChatBedrock(
        model_id=model_id,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 20,
        },
        region_name=region,
        config=config,
        streaming=False,
    )


def load_examples(
    path: Path,
    limit: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for _, row in read_jsonl(path):
        for field in (
            "prefix_text",
            "source_jsonl_line",
            "target_char_start",
        ):
            if row.get(field) is None:
                raise ValueError(
                    f"Frozen example is missing {field}"
                )

        rows.append(row)

        if limit > 0 and len(rows) >= limit:
            break

    if not rows:
        raise RuntimeError("No examples found")

    return rows


def add_target_words(
    examples: List[Dict[str, Any]],
    source_file: Path,
    text_field: str,
) -> None:
    by_line: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for example in examples:
        by_line[int(example["source_jsonl_line"])].append(example)

    required_lines = set(by_line)
    found_lines: set[int] = set()

    for line_number, source_row in read_jsonl(source_file):
        if line_number not in required_lines:
            continue

        text = source_row.get(text_field)

        if not isinstance(text, str):
            raise ValueError(
                f"{source_file}, line {line_number}: "
                f"{text_field!r} is not text"
            )

        for example in by_line[line_number]:
            example["_target_word"] = extract_next_unit(
                text,
                int(example["target_char_start"]),
            )

        found_lines.add(line_number)

        if found_lines == required_lines:
            break

    missing = sorted(required_lines - found_lines)

    if missing:
        raise RuntimeError(
            f"Source file is missing required lines: {missing[:10]}"
        )


def predict(model: ChatBedrock, prefix: str) -> str:
    prompt = (
        "Predict only the single immediate next word, number, or "
        "punctuation symbol after the Arabic legal-text prefix below. "
        "Return only that one unit. Do not explain.\n\n"
        f"{prefix}"
    )

    response = model.invoke(prompt)
    return first_unit(get_response_text(response))


def main() -> None:
    load_dotenv()
    args = parse_args()

    examples_path = Path(args.examples_file)
    source_path = Path(args.source_file)
    output_path = Path(args.output_file)

    if not examples_path.is_file():
        raise FileNotFoundError(examples_path)

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} exists; use --overwrite to replace it"
        )

    examples = load_examples(
        examples_path,
        args.limit,
    )

    add_target_words(
        examples,
        source_path,
        args.text_field,
    )

    model = create_model(
        args.model_id,
        args.region,
    )

    results: List[Dict[str, Any]] = []
    number_correct = 0

    for index, example in enumerate(examples, start=1):
        prefix = str(example["prefix_text"])
        target_word = str(example["_target_word"])
        predicted_word = predict(model, prefix)

        correct = normalize(predicted_word) == normalize(target_word)

        if correct:
            number_correct += 1

        results.append(
            {
                "prefix": prefix,
                "target_word": target_word,
                "predicted_word": predicted_word,
                "correct": correct,
            }
        )

        print(
            f"[{index}/{len(examples)}] "
            f"target={target_word!r} "
            f"predicted={predicted_word!r} "
            f"correct={correct}",
            flush=True,
        )

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    top1_accuracy = number_correct / len(results)

    output = {
        "model_id": args.model_id,
        "number_of_examples": len(results),
        "number_correct": number_correct,
        "top1_accuracy": top1_accuracy,
        "top1_percentage": top1_accuracy * 100,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            output,
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    print()
    print("TOP-1 NEXT-WORD EVALUATION")
    print("Examples:", len(results))
    print("Correct:", number_correct)
    print(f"Top-1 accuracy: {top1_accuracy:.4f}")
    print(f"Top-1 percentage: {top1_accuracy * 100:.2f}%")
    print("Output:", output_path)


if __name__ == "__main__":
    main()