import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

cache_root = Path(tempfile.gettempdir()) / "making-mlp-matplotlib-cache"
cache_root.mkdir(parents=True, exist_ok=True)
mpl_cache_dir = cache_root / "mpl"
xdg_cache_dir = cache_root / "xdg"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
xdg_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

from CONSTANTS import CONSTANTS
from lib.graph_drawer import graph_drawer


def _read_series(rows, *column_names):
    for column_name in column_names:
        if rows and column_name in rows[0]:
            return [
                float(row[column_name])
                for row in rows
                if row.get(column_name, "") != ""
            ]
    return []


def _remake_graph(project, timestamp):
    logs_path = Path(f"data/{project}/logs") / timestamp / "train_logs.csv"
    if not logs_path.exists():
        print(f"skip: missing {logs_path}")
        return False

    with open(logs_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"skip: empty {logs_path}")
        return False

    epochs = _read_series(rows, "epoch")
    train_losses = _read_series(rows, "train_loss", "total_loss")
    train_accuracies = _read_series(rows, "train_accuracy")
    valid_losses = _read_series(rows, "valid_loss")
    valid_accuracies = _read_series(rows, "valid_accuracy", "accuracy")

    graph_drawer(
        project,
        epochs,
        timestamp,
        train_losses,
        train_accuracies,
        valid_losses,
        valid_accuracies,
        show=False,
    )
    print(f"updated: data/{project}/logs/{timestamp}/train_graph.svg")
    return True


def _collect_timestamps(project, requested_timestamps):
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


def main():
    parser = argparse.ArgumentParser(description="Rebuild saved train graphs with the latest format.")
    parser.add_argument("--project", default=CONSTANTS.PROJECT, help="Target project under data/<project>/logs")
    parser.add_argument("--timestamp", action="append", dest="timestamps", help="Specific log timestamp to rebuild")
    args = parser.parse_args()

    timestamps = _collect_timestamps(args.project, args.timestamps)
    if not timestamps:
        print(f"no train logs found for project={args.project}")
        return

    updated_count = 0
    for timestamp in timestamps:
        updated_count += int(_remake_graph(args.project, timestamp))

    print(f"done: rebuilt {updated_count} graph(s)")


if __name__ == "__main__":
    main()
