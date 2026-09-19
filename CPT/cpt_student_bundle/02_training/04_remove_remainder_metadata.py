import json
import shutil
from collections import Counter
from pathlib import Path


PACKED_DIR = Path("data_preprocessing/data_cpt/final_cpt_v1/packed")

FILES = [
    PACKED_DIR / "train_cleanheaders_packed_4096.jsonl",
    PACKED_DIR / "val_cleanheaders_packed_4096.jsonl",
    PACKED_DIR / "test_cleanheaders_packed_4096.jsonl",
]

COLUMNS_TO_REMOVE = {
    "is_remainder_block",
}


def clean_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    backup_path = path.with_suffix(path.suffix + ".backup")
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    if not backup_path.exists():
        shutil.copy2(path, backup_path)
        print(f"Backup created: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")

    total_rows = 0
    changed_rows = 0

    with path.open("r", encoding="utf-8") as source, \
            temporary_path.open("w", encoding="utf-8") as destination:

        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error

            row_changed = False

            for column in COLUMNS_TO_REMOVE:
                if column in record:
                    record.pop(column)
                    row_changed = True

            if row_changed:
                changed_rows += 1

            destination.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            total_rows += 1

    temporary_path.replace(path)

    print(
        f"{path.name}: {total_rows:,} rows processed; "
        f"{changed_rows:,} rows changed."
    )


def validate_file(path: Path) -> None:
    schemas = Counter()

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            unexpected_columns = COLUMNS_TO_REMOVE.intersection(record.keys())

            if unexpected_columns:
                raise ValueError(
                    f"{path.name}, line {line_number}: "
                    f"columns still present: {sorted(unexpected_columns)}"
                )

            schemas[tuple(sorted(record.keys()))] += 1

    if len(schemas) != 1:
        print(f"\nWarning: {path.name} has {len(schemas)} schemas:")

        for schema, count in schemas.items():
            print(f"  {count:,} rows: {schema}")

        raise ValueError(
            f"{path.name} still has inconsistent columns."
        )

    schema = next(iter(schemas))

    print(f"{path.name}: validation successful.")
    print(f"Final columns: {schema}")


def main() -> None:
    print("Removing unused metadata columns...\n")

    for path in FILES:
        clean_file(path)
        validate_file(path)
        print()

    print("All packed files were cleaned successfully.")


if __name__ == "__main__":
    main()