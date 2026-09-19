import argparse
import json
from pathlib import Path
from contextlib import redirect_stdout

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="google/gemma-4-12B")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--report_file", type=str, required=True)

    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--append_eos", action="store_true")
    parser.add_argument("--drop_remainder", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    report_path = Path(args.report_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PACKING CPT DATASET")
    print("=" * 80)
    print("Model:", args.model_name)
    print("Input:", input_path)
    print("Output:", output_path)
    print("Report:", report_path)
    print("Max seq length:", args.max_seq_length)
    print("Append EOS:", args.append_eos)
    print("Drop remainder:", args.drop_remainder)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id. Cannot safely separate documents.")

    input_records = 0
    invalid_json = 0
    empty_text = 0

    total_input_tokens = 0
    total_output_tokens = 0
    output_blocks = 0

    buffer = []

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception:
                invalid_json += 1
                continue

            text = record.get("text", "") or ""

            if not text.strip():
                empty_text += 1
                continue

            input_records += 1

            ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            if args.append_eos:
                ids.append(tokenizer.eos_token_id)

            total_input_tokens += len(ids)
            buffer.extend(ids)

            while len(buffer) >= args.max_seq_length:
                block = buffer[:args.max_seq_length]
                buffer = buffer[args.max_seq_length:]

                out = {
                    "id": f"packed_{output_blocks:09d}",
                    "input_ids": block,
                    "attention_mask": [1] * len(block),
                    "packed": True,
                    "max_seq_length": args.max_seq_length,
                }

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")

                output_blocks += 1
                total_output_tokens += len(block)

        remainder_tokens = len(buffer)

        if not args.drop_remainder and remainder_tokens > 0:
            # For training speed we usually drop the remainder.
            # This option keeps it padded if needed.
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id

            attention_mask = [1] * remainder_tokens
            padding_needed = args.max_seq_length - remainder_tokens

            block = buffer + [pad_token_id] * padding_needed
            attention_mask = attention_mask + [0] * padding_needed

            out = {
                "id": f"packed_{output_blocks:09d}",
                "input_ids": block,
                "attention_mask": attention_mask,
                "packed": True,
                "max_seq_length": args.max_seq_length,
                "is_remainder_block": True,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            output_blocks += 1
            total_output_tokens += remainder_tokens

    dropped_tokens = total_input_tokens - total_output_tokens

    with report_path.open("w", encoding="utf-8") as report:
        with redirect_stdout(report):
            print("PACKED CPT DATASET REPORT")
            print("=" * 80)
            print("MODEL:", args.model_name)
            print("INPUT FILE:", input_path)
            print("OUTPUT FILE:", output_path)
            print("MAX_SEQ_LENGTH:", args.max_seq_length)
            print("APPEND_EOS:", args.append_eos)
            print("DROP_REMAINDER:", args.drop_remainder)

            print("\nINPUT QUALITY:")
            print("Input records:", input_records)
            print("Invalid JSON:", invalid_json)
            print("Empty text skipped:", empty_text)

            print("\nPACKING RESULT:")
            print("Input tokens including EOS:", total_input_tokens)
            print("Output blocks:", output_blocks)
            print("Output useful tokens:", total_output_tokens)
            print("Remainder tokens:", remainder_tokens)
            print("Dropped tokens:", dropped_tokens)

            if output_blocks > 0:
                print("Average tokens per packed block:", round(total_output_tokens / output_blocks, 2))

            print("\nEFFICIENCY:")
            print("Each full block has:", args.max_seq_length, "tokens")
            print("Approx optimizer steps with grad_accum=8:", round(output_blocks / 8, 2))

    print("Done.")
    print("Output:", output_path)
    print("Report:", report_path)
    print("Output blocks:", output_blocks)
    print("Input tokens:", total_input_tokens)
    print("Output useful tokens:", total_output_tokens)
    print("Dropped tokens:", dropped_tokens)


if __name__ == "__main__":
    main()