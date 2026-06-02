import argparse
import csv
import sys
from pathlib import Path
from statistics import median

sys.path.append(str(Path(__file__).resolve().parents[1]))

from CONSTANTS import CONSTANTS


REMOVED_COLUMNS = {
    "train_loss",
    "train_accuracy",
    "valid_loss",
    "valid_accuracy",
}

SUMMARY_METRIC_COLUMNS = [
    "train_loss_last10_avg",
    "train_loss_last10_median",
    "train_accuracy_last10_avg",
    "train_accuracy_last10_median",
    "valid_loss_last10_avg",
    "valid_loss_last10_median",
    "valid_accuracy_last10_avg",
    "valid_accuracy_last10_median",
]


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float_list(rows: list[dict[str, str]], column_name: str) -> list[float]:
    if not rows or column_name not in rows[0]:
        return []
    return [float(row[column_name]) for row in rows if row.get(column_name, "") != ""]


def _summarize_last10(rows: list[dict[str, str]], column_name: str) -> tuple[str, str]:
    values = _to_float_list(rows, column_name)
    if not values:
        return "", ""

    last_values = values[-10:]
    return (
        f"{sum(last_values) / len(last_values):.16g}",
        f"{median(last_values):.16g}",
    )


def _build_metrics(train_log_rows: list[dict[str, str]]) -> dict[str, str]:
    train_loss_avg, train_loss_median = _summarize_last10(train_log_rows, "train_loss")
    train_accuracy_avg, train_accuracy_median = _summarize_last10(train_log_rows, "train_accuracy")
    valid_loss_avg, valid_loss_median = _summarize_last10(train_log_rows, "valid_loss")
    valid_accuracy_avg, valid_accuracy_median = _summarize_last10(train_log_rows, "valid_accuracy")

    ep_times = _to_float_list(train_log_rows, "ep_time")
    req_time = f"{sum(ep_times):.16g}" if ep_times else ""
    avg_ep_time = f"{sum(ep_times) / len(ep_times):.16g}" if ep_times else ""

    metrics = {
        "NUM_EPOCHS": str(len(train_log_rows)) if train_log_rows else "",
        "req_time": req_time,
        "ep_time": avg_ep_time,
        "train_loss_last10_avg": train_loss_avg,
        "train_loss_last10_median": train_loss_median,
        "train_accuracy_last10_avg": train_accuracy_avg,
        "train_accuracy_last10_median": train_accuracy_median,
        "valid_loss_last10_avg": valid_loss_avg,
        "valid_loss_last10_median": valid_loss_median,
        "valid_accuracy_last10_avg": valid_accuracy_avg,
        "valid_accuracy_last10_median": valid_accuracy_median,
    }
    return metrics


def _build_row(project: str, timestamp: str, existing_row: dict[str, str], train_log_rows: list[dict[str, str]]) -> dict[str, str]:
    row = dict(existing_row)
    row.setdefault("PROJECT", project)
    row["time_stamp"] = timestamp

    for column in REMOVED_COLUMNS:
        row.pop(column, None)

    row.update(_build_metrics(train_log_rows))
    return row


def _collect_timestamps(project: str, requested_timestamps: list[str] | None) -> list[str]:
    if requested_timestamps:
        return requested_timestamps

    logs_root = Path(f"data/{project}/logs")
    if not logs_root.exists():
        return []

    return sorted(
        path.name
        for path in logs_root.iterdir()
        if path.is_dir() and (path / "train_logs.csv").exists()
    )


def _build_header(existing_rows: list[dict[str, str]], remade_rows: list[dict[str, str]]) -> list[str]:
    header: list[str] = []

    for row in existing_rows:
        for column in row.keys():
            if column not in REMOVED_COLUMNS and column not in header:
                header.append(column)

    for row in remade_rows:
        for column in row.keys():
            if column not in REMOVED_COLUMNS and column not in header:
                header.append(column)

    for column in SUMMARY_METRIC_COLUMNS:
        if column not in header:
            header.append(column)

    return header


def main():
    parser = argparse.ArgumentParser(description="Rebuild summary logs using last-10-epoch aggregates.")
    parser.add_argument("--project", default=CONSTANTS.PROJECT, help="Target project under data/<project>/logs")
    parser.add_argument("--timestamp", action="append", dest="timestamps", help="Specific log timestamp to rebuild")
    args = parser.parse_args()

    summary_path = Path(f"data/{args.project}/logs/{args.project}.csv")
    existing_rows = _read_csv_rows(summary_path)
    existing_by_timestamp = {
        row["time_stamp"]: row for row in existing_rows if row.get("time_stamp")
    }

    timestamps = _collect_timestamps(args.project, args.timestamps)
    if not timestamps:
        print(f"no train logs found for project={args.project}")
        return

    remade_by_timestamp: dict[str, dict[str, str]] = {}
    for timestamp in timestamps:
        train_logs_path = Path(f"data/{args.project}/logs/{timestamp}/train_logs.csv")
        train_log_rows = _read_csv_rows(train_logs_path)
        if not train_log_rows:
            print(f"skip: empty {train_logs_path}")
            continue

        existing_row = existing_by_timestamp.get(timestamp, {})
        remade_by_timestamp[timestamp] = _build_row(args.project, timestamp, existing_row, train_log_rows)

    output_rows: list[dict[str, str]] = []
    seen_timestamps = set()
    for row in existing_rows:
        timestamp = row.get("time_stamp", "")
        if timestamp in remade_by_timestamp:
            output_rows.append(remade_by_timestamp[timestamp])
            seen_timestamps.add(timestamp)
            continue

        preserved_row = {key: value for key, value in row.items() if key not in REMOVED_COLUMNS}
        output_rows.append(preserved_row)
        if timestamp:
            seen_timestamps.add(timestamp)

    for timestamp in sorted(remade_by_timestamp):
        if timestamp not in seen_timestamps:
            output_rows.append(remade_by_timestamp[timestamp])

    header = _build_header(existing_rows, output_rows)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({column: row.get(column, "") for column in header})

    print(f"updated: {summary_path}")
    print(f"done: rebuilt {len(remade_by_timestamp)} summary row(s)")


if __name__ == "__main__":
    main()
