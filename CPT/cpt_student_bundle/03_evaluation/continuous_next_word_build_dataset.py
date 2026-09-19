#!/usr/bin/env python3
"""Build a deterministic rolling continuous complete-next-word benchmark."""

import argparse
import hashlib
import json
import random
import unicodedata
from pathlib import Path

TEXT_FIELDS = ("text", "clean_text", "content", "document_text", "body")
ID_FIELDS = ("id", "document_id", "doc_id", "source_id")
SOURCE_FIELDS = ("source", "source_name", "source_type", "collection", "dataset", "origin")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_field(row, requested, candidates, required=False):
    if requested != "auto":
        if requested not in row:
            raise KeyError(
                f"Field {requested!r} not found. Available: {sorted(row)}"
            )
        return requested
    for field in candidates:
        if field in row:
            return field
    if required:
        raise KeyError(
            f"Could not detect required field. Available: {sorted(row)}"
        )
    return None


def word_spans(text):
    """Unicode letter/number words; punctuation stays in context."""
    spans = []
    start = None
    for i, ch in enumerate(text):
        major = unicodedata.category(ch)[0]
        is_word = major in {"L", "N"} or (major == "M" and start is not None)
        if is_word:
            if start is None:
                start = i
        elif start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    return spans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--text_field", default="auto")
    p.add_argument("--id_field", default="auto")
    p.add_argument("--source_field", default="auto")
    p.add_argument("--documents", type=int, default=20)
    p.add_argument("--predictions_per_document", type=int, default=50)
    p.add_argument("--warmup_words", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument(
        "--context_boundary",
        choices=("word_end", "separator"),
        default="word_end",
        help=(
            "word_end (v2, default): the context ends flush on the last "
            "character of the previous word, so the model generates the "
            "separator and the next word naturally (matches Gemma's "
            "space-attached token vocabulary). separator (v1): the context "
            "runs up to the target's first character, so it ends with the "
            "trailing space/newline — kept only to reproduce the v1 "
            "benchmark byte-for-byte."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    source = Path(args.source_file).expanduser().resolve()
    output = Path(args.output_file).expanduser().resolve()
    meta = Path(str(output) + ".meta.json")

    if not source.is_file():
        raise SystemExit(f"Source not found: {source}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output}; use --overwrite")

    needed = args.warmup_words + args.predictions_per_document
    eligible = []
    text_field = id_field = source_field = None

    with source.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)

            if text_field is None:
                text_field = detect_field(
                    row, args.text_field, TEXT_FIELDS, required=True
                )
                id_field = detect_field(row, args.id_field, ID_FIELDS)
                source_field = detect_field(
                    row, args.source_field, SOURCE_FIELDS
                )

            text = row.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue

            spans = word_spans(text)
            if len(spans) < needed:
                continue

            eligible.append(
                {
                    "line": line_no,
                    "id": (
                        str(row.get(id_field))
                        if id_field and row.get(id_field) is not None
                        else f"source-line-{line_no}"
                    ),
                    "source": (
                        str(row.get(source_field)).strip()
                        if source_field
                        and row.get(source_field) is not None
                        and str(row.get(source_field)).strip()
                        else (
                            str(row.get(id_field)).split("__", 1)[0].strip()
                            if id_field
                            and row.get(id_field) is not None
                            and "__" in str(row.get(id_field))
                            else "unknown"
                        )
                    ),
                    "text": text,
                    "spans": spans,
                }
            )

    if len(eligible) < args.documents:
        raise SystemExit(
            f"Only {len(eligible)} eligible documents; "
            f"{args.documents} requested."
        )

    rng = random.Random(args.seed)

    # Select deterministically across available source collections so that
    # one large source does not dominate the benchmark by chance.
    buckets = {}
    for document in eligible:
        buckets.setdefault(document["source"], []).append(document)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    sources = sorted(buckets)
    rng.shuffle(sources)
    selected = []
    while len(selected) < args.documents:
        progressed = False
        for source_name in sources:
            if buckets[source_name] and len(selected) < args.documents:
                selected.append(buckets[source_name].pop())
                progressed = True
        if not progressed:
            break

    if len(selected) != args.documents:
        raise SystemExit("Could not select the requested documents.")

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    selected_meta = []

    with output.open("w", encoding="utf-8") as out:
        for doc_no, doc in enumerate(selected, 1):
            max_start = len(doc["spans"]) - args.predictions_per_document
            first_target = rng.randint(args.warmup_words, max_start)
            context_first = first_target - args.warmup_words
            context_start_char = doc["spans"][context_first][0]
            passage_id = f"passage-{doc_no:03d}"

            selected_meta.append(
                {
                    "passage_id": passage_id,
                    "source_jsonl_line": doc["line"],
                    "document_id": doc["id"],
                    "source": doc["source"],
                    "first_target_word_index": first_target,
                }
            )

            for offset in range(args.predictions_per_document):
                target_index = first_target + offset
                start, end = doc["spans"][target_index]

                # v2 (word_end): the context ends flush on the previous
                # word so the model generates the separator + target
                # naturally. v1 (separator): the context includes the
                # trailing separator up to the target's first character.
                if args.context_boundary == "word_end":
                    context_end = doc["spans"][target_index - 1][1]
                else:
                    context_end = start

                count += 1

                record = {
                    "example_id": f"rolling-{count:06d}",
                    "passage_id": passage_id,
                    "position_in_passage": offset + 1,
                    "source_jsonl_line": doc["line"],
                    "document_id": doc["id"],
                    "source": doc["source"],
                    "context_word_count": args.warmup_words + offset,
                    "context": doc["text"][context_start_char:context_end],
                    "context_boundary": args.context_boundary,
                    "context_end_char": context_end,
                    "target_word": doc["text"][start:end],
                    "target_char_start": start,
                    "target_char_end": end,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "status": "completed",
        "benchmark_type": "rolling_continuous_complete_next_word",
        "source_file": str(source),
        "source_file_sha256": sha256_file(source),
        "output_file": str(output),
        "output_file_sha256": sha256_file(output),
        "seed": args.seed,
        "text_field": text_field,
        "id_field": id_field,
        "source_field": source_field,
        "eligible_documents": len(eligible),
        "eligible_source_distribution": {
            source_name: sum(1 for d in eligible if d["source"] == source_name)
            for source_name in sorted({d["source"] for d in eligible})
        },
        "selected_source_distribution": {
            source_name: sum(1 for d in selected if d["source"] == source_name)
            for source_name in sorted({d["source"] for d in selected})
        },
        "number_of_documents": args.documents,
        "warmup_words": args.warmup_words,
        "predictions_per_document": args.predictions_per_document,
        "number_of_examples": count,
        "context_boundary": args.context_boundary,
        "context_policy": (
            "Rolling true-text context; each next case includes the true "
            "preceding word, never the model prediction. "
            + (
                "v2 boundary: the context ends flush on the previous "
                "word; the model must generate the separator and the "
                "target word itself."
                if args.context_boundary == "word_end"
                else "v1 boundary: the context includes the trailing "
                "separator up to the target's first character."
            )
        ),
        "selected_documents": selected_meta,
    }
    meta.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Created:", output)
    print("Metadata:", meta)
    print("Text field:", text_field)
    print("Documents:", args.documents)
    print("Selected sources:")
    for source_name, n in metadata["selected_source_distribution"].items():
        print(f"  {source_name}: {n}")
    print("Examples:", count)
    print("Benchmark SHA-256:", metadata["output_file_sha256"])


if __name__ == "__main__":
    main()