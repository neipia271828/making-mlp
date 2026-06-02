import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path

LEGACY_COLUMN_MAP = {
    "model": "MODEL",
    "epochs": "NUM_EPOCHS",
}


def _dataclass_to_dict(config) -> dict[str, object]:
    if not is_dataclass(config):
        raise TypeError(f"Expected dataclass instance, got {type(config).__name__}")
    return asdict(config)


def _normalize_row_keys(row: dict[str, object]) -> dict[str, object]:
    normalized_row = dict(row)
    for old_key, new_key in LEGACY_COLUMN_MAP.items():
        if old_key in normalized_row and new_key not in normalized_row:
            normalized_row[new_key] = normalized_row.pop(old_key)
    return normalized_row


def _normalize_column_name(column: str) -> str:
    stripped_column = column.strip()
    return LEGACY_COLUMN_MAP.get(stripped_column, stripped_column)


def _read_rows_with_normalized_header(csv_path: Path) -> tuple[list[str], list[dict[str, object]]]:
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        raw_header = next(reader, [])
        normalized_header = [_normalize_column_name(column) for column in raw_header]
        rows: list[dict[str, object]] = []

        for values in reader:
            row: dict[str, object] = {}
            for index, column in enumerate(normalized_header):
                value = values[index] if index < len(values) else ""
                if column not in row or row[column] == "":
                    row[column] = value
            rows.append(row)

    unique_header = list(dict.fromkeys(normalized_header))
    return unique_header, rows


def _rewrite_csv_with_new_header(csv_path: Path, header: list[str]) -> None:
    _, rows = _read_rows_with_normalized_header(csv_path)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in header})


def make_summalize_csv(project: str, meta_constants, model_constants, metrics: dict[str, object]):
    row = {
        **_dataclass_to_dict(meta_constants),
        **_dataclass_to_dict(model_constants),
        **metrics,
    }
    row = _normalize_row_keys(row)
    header = list(row.keys())

    csv_path = Path(".") / f"data/{project}/logs/{project}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerow(row)
        return

    existing_header, _ = _read_rows_with_normalized_header(csv_path)

    merged_header = list(header)
    for column in existing_header:
        if column in LEGACY_COLUMN_MAP:
            continue
        if column not in merged_header:
            merged_header.append(column)

    if merged_header != existing_header:
        _rewrite_csv_with_new_header(csv_path, merged_header)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_header or header)
        writer.writerow(row)

def make_train_log(PROJECT, timestamp, logs):

    logs_dir = Path(f"data/{PROJECT}/logs") / timestamp
    logs_dir.mkdir(parents=True, exist_ok=True)

    logs_path = logs_dir / "train_logs.csv"

    with open(logs_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ep_time",
            "epoch",
            "train_loss",
            "train_accuracy",
            "valid_loss",
            "valid_accuracy",
        ])

        for l in logs:
            writer.writerow(l)
