"""
Document-level train/val/test split for the Lebanese legal CPT corpus.

Replaces 03_split_train_val_test_ORIGINAL.py, which shuffled *records* grouped by
*source*. Records are chunks produced with OVERLAP_CHARS = 1200, so two chunks of one
document landing on opposite sides of the split share 1200 characters of literal text.
That deflates held-out perplexity for a reason unrelated to learning.

Three changes from the original (Week2_CPT_Plan.md, Part 1.1):
  1. group by document, not by source
  2. ratios 90/5/5, not 97/2/1
  3. per-source floors max(1, ...), not max(4, ...) / max(20, ...)

On (1): the plan says to group by `source_id`. Measured on the delivered subset, 396
`source_id` values are used by two different sources (bibliographic_rulings and
related_provisions number their records from the same range), and none of those pairs
are the same document. Grouping by bare `source_id` would therefore fuse 396 pairs of
unrelated documents into groups spanning two sources, which makes the per-source
stratification below ambiguous. The document key is the composite:

    (final_cpt_source, source_id)

which yields 43,680 documents rather than the 43,284 quoted in the plan. It also keeps
every piece of a document together, including the second-level `token_split_4096_index`
pieces that make `chunk_index` repeat within a document.

On (3): associated_studies_ar has only 7 documents in this subset. The original floors
would put 4 in val and 2 in test, leaving 1 for training, and the source would
effectively vanish from the training data.

Usage:
    python 03b_split_by_document.py \
        --input  data/legal_corpus_subset.jsonl \
        --outdir splits \
        --report reports/03b_split_by_document.txt
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

SEED = 42


def get_source(record):
    return record.get("final_cpt_source") or record.get("source") or "unknown"


def get_doc_key(record):
    """The composite document key. See module docstring for why `source_id` alone is not enough."""
    source_id = record.get("source_id")
    if source_id is None or str(source_id).strip() == "":
        raise SystemExit(
            "FATAL: record {!r} has no source_id. "
            "The document-level split cannot be built without it.".format(record.get("id"))
        )
    return (get_source(record), str(source_id))


def estimate_tokens_from_text(text):
    # Same estimator as subsample_corpus.py and the original split script, so the
    # numbers in this report are comparable to the supervisor's.
    return round(len(text) / 2.6)


def summarize(records):
    source_counts = Counter()
    source_est_tokens = Counter()
    source_docs = defaultdict(set)

    total_chars = 0
    total_est_tokens = 0

    for r in records:
        source = get_source(r)
        text = r.get("text", "") or ""
        est = estimate_tokens_from_text(text)

        source_counts[source] += 1
        source_est_tokens[source] += est
        source_docs[source].add(get_doc_key(r))

        total_chars += len(text)
        total_est_tokens += est

    return {
        "records": len(records),
        "documents": sum(len(v) for v in source_docs.values()),
        "total_chars": total_chars,
        "total_est_tokens": total_est_tokens,
        "source_counts": source_counts,
        "source_est_tokens": source_est_tokens,
        "source_docs": source_docs,
    }


def measure_record_level_leakage(by_source_docs, rng_seed):
    """
    Counterfactual: what the ORIGINAL record-level split would have leaked on this
    subset. Reproduces the plan's leakage table so the improvement is measurable
    rather than asserted. Reported only, never acted on.
    """
    rng = random.Random(rng_seed)
    train, val, test = [], [], []

    for source in sorted(by_source_docs):
        records = [r for doc_records in by_source_docs[source].values() for r in doc_records]
        rng.shuffle(records)
        n = len(records)
        test_n = max(1, round(n * TEST_RATIO))
        val_n = max(1, round(n * VAL_RATIO))
        test.extend(records[:test_n])
        val.extend(records[test_n:test_n + val_n])
        train.extend(records[test_n + val_n:])

    train_docs = {get_doc_key(r) for r in train}
    out = {}
    for name, split in (("val", val), ("test", test)):
        leaked = sum(1 for r in split if get_doc_key(r) in train_docs)
        out[name] = (leaked, len(split))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/legal_corpus_subset.jsonl")
    ap.add_argument("--outdir", default="splits")
    ap.add_argument("--report", default="reports/03b_split_by_document.txt")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--max_train_tokens",
        type=int,
        default=0,
        help="If >0, take a further DOCUMENT-level slice of the train split only, "
             "down to roughly this many estimated tokens. val and test are never sliced.",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    report_path = Path(args.report)
    outdir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = outdir / "train_doclevel.jsonl"
    val_path = outdir / "val_doclevel.jsonl"
    test_path = outdir / "test_doclevel.jsonl"

    rng = random.Random(args.seed)

    # ---- load, grouped source -> doc_key -> [records] ---------------------------
    by_source_docs = defaultdict(lambda: defaultdict(list))
    input_records = 0
    invalid_json = []
    empty_text = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as e:
                invalid_json.append((line_number, str(e)))
                continue

            text = record.get("text", "") or ""
            if not text.strip():
                empty_text.append((line_number, record.get("id")))
                continue

            by_source_docs[get_source(record)][get_doc_key(record)].append(record)
            input_records += 1

    leakage = measure_record_level_leakage(by_source_docs, args.seed)

    # ---- assign WHOLE DOCUMENTS, stratified per source --------------------------
    train_records, val_records, test_records = [], [], []
    split_plan = []

    for source in sorted(by_source_docs):
        docs = by_source_docs[source]
        doc_keys = sorted(docs)          # sort first so the shuffle is reproducible
        rng.shuffle(doc_keys)

        n = len(doc_keys)
        test_n = max(1, round(n * TEST_RATIO))
        val_n = max(1, round(n * VAL_RATIO))

        # Do not consume a small source entirely into val/test. associated_studies_ar
        # has 7 documents here, so this is the branch that keeps 5 of them in train.
        if val_n + test_n >= n:
            test_n = 1
            val_n = 1

        test_keys = doc_keys[:test_n]
        val_keys = doc_keys[test_n:test_n + val_n]
        train_keys = doc_keys[test_n + val_n:]

        for k in train_keys:
            train_records.extend(docs[k])
        for k in val_keys:
            val_records.extend(docs[k])
        for k in test_keys:
            test_records.extend(docs[k])

        split_plan.append((source, n, len(train_keys), len(val_keys), len(test_keys)))

    # ---- optional further document-level slice of TRAIN only --------------------
    train_sliced_from = None
    if args.max_train_tokens > 0:
        train_by_doc = defaultdict(list)
        for r in train_records:
            train_by_doc[get_doc_key(r)].append(r)

        total_tokens = sum(
            estimate_tokens_from_text(r.get("text", "") or "") for r in train_records
        )
        train_sliced_from = total_tokens
        frac = min(1.0, args.max_train_tokens / max(1, total_tokens))

        # Slice proportionally within each source so the source mix is preserved.
        by_src = defaultdict(list)
        for k in train_by_doc:
            by_src[k[0]].append(k)

        keep_keys = []
        for source in sorted(by_src):
            keys = sorted(by_src[source])
            rng.shuffle(keys)
            keep_keys.extend(keys[: max(1, round(len(keys) * frac))])

        train_records = [r for k in keep_keys for r in train_by_doc[k]]

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    # ---- acceptance checks (Week2_CPT_Plan.md, Part 1.1) ------------------------
    train_docs = {get_doc_key(r) for r in train_records}
    val_docs = {get_doc_key(r) for r in val_records}
    test_docs = {get_doc_key(r) for r in test_records}

    # 1. No document appears in two splits.
    assert not (train_docs & val_docs), "{} docs in both train and val".format(len(train_docs & val_docs))
    assert not (train_docs & test_docs), "{} docs in both train and test".format(len(train_docs & test_docs))
    assert not (val_docs & test_docs), "{} docs in both val and test".format(len(val_docs & test_docs))

    # 2. Every source appears in all three splits.
    sources = set(by_source_docs)
    for split_name, split in (("train", train_records), ("val", val_records), ("test", test_records)):
        present = {get_source(r) for r in split}
        missing = sources - present
        assert not missing, "{} is missing sources: {}".format(split_name, missing)

    # 3. No source lost its training data to a floor.
    train_per_source = Counter(get_source(r) for r in train_records)
    for s in sources:
        n = train_per_source[s]
        assert n >= 3, "source {} has only {} training records".format(s, n)

    # 4. Chunk integrity: every document is written whole, into exactly one split.
    #    A sliced train split legitimately drops whole documents, so only the
    #    documents actually present are checked.
    for split_name, split in (("train", train_records), ("val", val_records), ("test", test_records)):
        counts = Counter(get_doc_key(r) for r in split)
        for k, actual in counts.items():
            expected = len(by_source_docs[k[0]][k])
            assert actual == expected, "{}: document {} has {} of {} records".format(
                split_name, k, actual, expected
            )

    def write_jsonl(path, records):
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    write_jsonl(test_path, test_records)

    train_summary = summarize(train_records)
    val_summary = summarize(val_records)
    test_summary = summarize(test_records)
    all_summary = summarize(train_records + val_records + test_records)
    total_written = train_summary["records"] + val_summary["records"] + test_summary["records"]
    total_tokens = (
        train_summary["total_est_tokens"]
        + val_summary["total_est_tokens"]
        + test_summary["total_est_tokens"]
    )

    def emit():
        print("DOCUMENT-LEVEL TRAIN / VALIDATION / TEST SPLIT REPORT")
        print("=" * 80)
        print("INPUT FILE:", input_path)
        print("TRAIN FILE:", train_path)
        print("VAL FILE:", val_path)
        print("TEST FILE:", test_path)
        print("TRAIN_RATIO:", TRAIN_RATIO)
        print("VAL_RATIO:", VAL_RATIO)
        print("TEST_RATIO:", TEST_RATIO)
        print("SEED:", args.seed)
        print("DOCUMENT KEY: (final_cpt_source, source_id)")
        if train_sliced_from is not None:
            print("TRAIN SLICED: from {:,} to <= {:,} est. tokens (document-level)".format(
                train_sliced_from, args.max_train_tokens))

        print("\nINPUT QUALITY:")
        print("Input valid records:", input_records)
        print("Invalid JSON:", len(invalid_json))
        print("Empty text skipped:", len(empty_text))

        print("\nLEAKAGE AVOIDED (counterfactual record-level split, same ratios/seed):")
        for name, (leaked, total) in leakage.items():
            pct = leaked / total * 100 if total else 0.0
            print("  {}: {} of {} held-out records ({:.1f}%) would share a document with train".format(
                name, leaked, total, pct))
        print("  document-level split: 0 (asserted above)")

        print("\nFINAL COUNTS:")
        for name, s in (("Train", train_summary), ("Validation", val_summary), ("Test", test_summary)):
            print("{} documents: {}  records: {}  est_tokens: {:,}".format(
                name, s["documents"], s["records"], s["total_est_tokens"]))
        print("Total written:", total_written)

        print("\nFINAL RATIOS (by record % / by est. token %):")
        for name, s in (("Train", train_summary), ("Validation", val_summary), ("Test", test_summary)):
            print("{}: {} / {}".format(
                name,
                round(s["records"] / total_written * 100, 3),
                round(s["total_est_tokens"] / total_tokens * 100, 3)))

        print("\nSPLIT PLAN BY SOURCE (documents):")
        for source, total, tr, va, te in sorted(split_plan):
            print("  {:<28} total: {:>6}  train: {:>6}  val: {:>5}  test: {:>5}".format(
                source, total, tr, va, te))

        for name, s in (("TRAIN", train_summary), ("VALIDATION", val_summary), ("TEST", test_summary)):
            print("\nSOURCE COUNTS - {}:".format(name))
            for source, count in s["source_counts"].most_common():
                print("  {:<28} docs: {:>6}  records: {:>6}  est_tokens: {:>12,}".format(
                    source, len(s["source_docs"][source]), count, s["source_est_tokens"][source]))

        print("\nTOTAL WRITTEN SUMMARY:")
        print("Records:", all_summary["records"])
        print("Documents:", all_summary["documents"])
        print("Chars: {:,}".format(all_summary["total_chars"]))
        print("Estimated tokens: {:,}".format(all_summary["total_est_tokens"]))

        print("\nINVALID JSON EXAMPLES:")
        for item in invalid_json[:20]:
            print(item)

        print("\nEMPTY TEXT EXAMPLES:")
        for item in empty_text[:20]:
            print(item)

    with report_path.open("w", encoding="utf-8") as report:
        with redirect_stdout(report):
            emit()
    emit()

    print("\nAll acceptance checks passed.")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
