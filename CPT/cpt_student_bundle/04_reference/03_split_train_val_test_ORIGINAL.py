import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from contextlib import redirect_stdout

INPUT_PATH = Path("data_cpt/cpt_full_legal_v1_4096.jsonl")

TRAIN_PATH = Path("data_cpt/train_cpt_full_legal_v1_4096.jsonl")
VAL_PATH = Path("data_cpt/val_cpt_full_legal_v1_4096.jsonl")
TEST_PATH = Path("data_cpt/test_cpt_full_legal_v1_4096.jsonl")

REPORT_PATH = Path("reports/28_train_val_test_split_4096.txt")

TRAIN_RATIO = 0.97
VAL_RATIO = 0.02
TEST_RATIO = 0.01

SEED = 42

TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
VAL_PATH.parent.mkdir(parents=True, exist_ok=True)
TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

random.seed(SEED)


def get_source(record):
    return record.get("final_cpt_source") or record.get("source") or "unknown"


def estimate_tokens_from_text(text):
    # Quick report estimate only.
    # Exact Gemma token check was already done before.
    return round(len(text) / 2.6)


groups = defaultdict(list)

input_records = 0
invalid_json = []
empty_text = []

with INPUT_PATH.open("r", encoding="utf-8") as f:
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

        source = get_source(record)
        groups[source].append(record)
        input_records += 1


train_records = []
val_records = []
test_records = []

split_plan = []

for source, records in groups.items():
    random.shuffle(records)

    n = len(records)

    test_n = max(1, round(n * TEST_RATIO))
    val_n = max(1, round(n * VAL_RATIO))

    # Ensure small but important sources appear in val/test
    if source == "associated_studies_ar":
        val_n = max(4, val_n)
        test_n = max(2, test_n)

    if source == "related_provisions":
        val_n = max(20, val_n)
        test_n = max(10, test_n)

    # Safety: do not consume all records into val/test
    if val_n + test_n >= n:
        test_n = max(1, round(n * TEST_RATIO))
        val_n = max(1, n - test_n - 1)

    test_part = records[:test_n]
    val_part = records[test_n:test_n + val_n]
    train_part = records[test_n + val_n:]

    test_records.extend(test_part)
    val_records.extend(val_part)
    train_records.extend(train_part)

    split_plan.append(
        (
            source,
            n,
            len(train_part),
            len(val_part),
            len(test_part),
        )
    )


random.shuffle(train_records)
random.shuffle(val_records)
random.shuffle(test_records)


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


write_jsonl(TRAIN_PATH, train_records)
write_jsonl(VAL_PATH, val_records)
write_jsonl(TEST_PATH, test_records)


def summarize(records):
    source_counts = Counter()
    source_chars = Counter()
    source_est_tokens = Counter()

    total_chars = 0
    total_est_tokens = 0

    split_child_records = 0
    parent_counter = Counter()

    for r in records:
        source = get_source(r)
        text = r.get("text", "") or ""

        chars = len(text)
        est_tokens = estimate_tokens_from_text(text)

        source_counts[source] += 1
        source_chars[source] += chars
        source_est_tokens[source] += est_tokens

        total_chars += chars
        total_est_tokens += est_tokens

        parent_id = r.get("parent_id_before_token_split")
        if parent_id:
            split_child_records += 1
            parent_counter[parent_id] += 1

    return {
        "records": len(records),
        "total_chars": total_chars,
        "total_est_tokens": total_est_tokens,
        "source_counts": source_counts,
        "source_chars": source_chars,
        "source_est_tokens": source_est_tokens,
        "split_child_records": split_child_records,
        "split_parent_count": len(parent_counter),
    }


train_summary = summarize(train_records)
val_summary = summarize(val_records)
test_summary = summarize(test_records)
all_summary = summarize(train_records + val_records + test_records)


with REPORT_PATH.open("w", encoding="utf-8") as report:
    with redirect_stdout(report):
        print("TRAIN / VALIDATION / TEST SPLIT REPORT - 4096 DATASET")
        print("=" * 80)
        print("INPUT FILE:", INPUT_PATH)
        print("TRAIN FILE:", TRAIN_PATH)
        print("VAL FILE:", VAL_PATH)
        print("TEST FILE:", TEST_PATH)
        print("TRAIN_RATIO:", TRAIN_RATIO)
        print("VAL_RATIO:", VAL_RATIO)
        print("TEST_RATIO:", TEST_RATIO)
        print("SEED:", SEED)

        print("\nINPUT QUALITY:")
        print("Input valid records:", input_records)
        print("Invalid JSON:", len(invalid_json))
        print("Empty text skipped:", len(empty_text))

        print("\nFINAL COUNTS:")
        print("Train records:", len(train_records))
        print("Validation records:", len(val_records))
        print("Test records:", len(test_records))
        print("Total written:", len(train_records) + len(val_records) + len(test_records))

        print("\nFINAL RATIOS:")
        total_written = len(train_records) + len(val_records) + len(test_records)
        print("Train %:", round(len(train_records) / total_written * 100, 4))
        print("Validation %:", round(len(val_records) / total_written * 100, 4))
        print("Test %:", round(len(test_records) / total_written * 100, 4))

        print("\nSPLIT PLAN BY SOURCE:")
        for source, total, train_n, val_n, test_n in sorted(split_plan):
            print(
                source,
                "total:", total,
                "train:", train_n,
                "val:", val_n,
                "test:", test_n,
            )

        print("\nTRAIN SUMMARY:")
        print("Records:", train_summary["records"])
        print("Chars:", train_summary["total_chars"])
        print("Estimated tokens:", train_summary["total_est_tokens"])
        print("Split child records:", train_summary["split_child_records"])
        print("Split parent count:", train_summary["split_parent_count"])

        print("\nVALIDATION SUMMARY:")
        print("Records:", val_summary["records"])
        print("Chars:", val_summary["total_chars"])
        print("Estimated tokens:", val_summary["total_est_tokens"])
        print("Split child records:", val_summary["split_child_records"])
        print("Split parent count:", val_summary["split_parent_count"])

        print("\nTEST SUMMARY:")
        print("Records:", test_summary["records"])
        print("Chars:", test_summary["total_chars"])
        print("Estimated tokens:", test_summary["total_est_tokens"])
        print("Split child records:", test_summary["split_child_records"])
        print("Split parent count:", test_summary["split_parent_count"])

        print("\nSOURCE COUNTS - TRAIN:")
        for source, count in train_summary["source_counts"].most_common():
            print(
                source,
                "records:", count,
                "chars:", train_summary["source_chars"][source],
                "est_tokens:", train_summary["source_est_tokens"][source],
            )

        print("\nSOURCE COUNTS - VALIDATION:")
        for source, count in val_summary["source_counts"].most_common():
            print(
                source,
                "records:", count,
                "chars:", val_summary["source_chars"][source],
                "est_tokens:", val_summary["source_est_tokens"][source],
            )

        print("\nSOURCE COUNTS - TEST:")
        for source, count in test_summary["source_counts"].most_common():
            print(
                source,
                "records:", count,
                "chars:", test_summary["source_chars"][source],
                "est_tokens:", test_summary["source_est_tokens"][source],
            )

        print("\nTOTAL WRITTEN SUMMARY:")
        print("Records:", all_summary["records"])
        print("Chars:", all_summary["total_chars"])
        print("Estimated tokens:", all_summary["total_est_tokens"])

        print("\nINVALID JSON EXAMPLES:")
        for item in invalid_json[:20]:
            print(item)

        print("\nEMPTY TEXT EXAMPLES:")
        for item in empty_text[:20]:
            print(item)


print("Done.")
print("Train file:", TRAIN_PATH)
print("Validation file:", VAL_PATH)
print("Test file:", TEST_PATH)
print("Report:", REPORT_PATH)
print("Train records:", len(train_records))
print("Validation records:", len(val_records))
print("Test records:", len(test_records))