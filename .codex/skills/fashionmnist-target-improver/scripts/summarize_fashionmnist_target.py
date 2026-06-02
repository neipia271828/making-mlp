#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize FashionMNIST logs against a target.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--logs-dir", required=True)
    return parser.parse_args()


def read_target_accuracy(target_path: Path) -> float | None:
    text = target_path.read_text(encoding="utf-8")
    match = re.search(r"accuracy\s*[-:]\s*>=?\s*([0-9]+(?:\.[0-9]+)?)%?", text, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    return value / 100 if value > 1 else value


def to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def read_rows(csv_path: Path) -> list[dict[str, object]]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    normalized = []
    for row in rows:
        normalized.append(
            {
                **row,
                "NUM_EPOCHS": int(row["NUM_EPOCHS"]) if row.get("NUM_EPOCHS") else None,
                "L_LATE": to_float(row.get("L_LATE", "")),
                "WEIGHT_DECAY": to_float(row.get("WEIGHT_DECAY", "")),
                "valid_accuracy_last10_avg": to_float(row.get("valid_accuracy_last10_avg", "")),
                "valid_accuracy_last10_median": to_float(row.get("valid_accuracy_last10_median", "")),
                "valid_loss_last10_avg": to_float(row.get("valid_loss_last10_avg", "")),
                "train_accuracy_last10_avg": to_float(row.get("train_accuracy_last10_avg", "")),
            }
        )
    return normalized


def best_rows(rows: list[dict[str, object]], limit: int = 5) -> list[dict[str, object]]:
    scored = [row for row in rows if row.get("valid_accuracy_last10_avg") is not None]
    return sorted(scored, key=lambda row: row["valid_accuracy_last10_avg"], reverse=True)[:limit]


def config_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("MODEL"),
        row.get("L_LATE"),
        row.get("WEIGHT_DECAY"),
        row.get("SCHEDULER_NAME", ""),
        row.get("ETA_MIN", ""),
    )


def inspect_run_trend(logs_dir: Path, timestamp: str) -> dict[str, float | None]:
    train_log_path = logs_dir / timestamp / "train_logs.csv"
    if not train_log_path.exists():
        return {"recent_valid_acc_delta": None, "recent_valid_loss_delta": None}

    with train_log_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    valid_acc = [float(row["valid_accuracy"]) for row in rows if row.get("valid_accuracy")]
    valid_loss = [float(row["valid_loss"]) for row in rows if row.get("valid_loss")]

    if len(valid_acc) < 10 or len(valid_loss) < 10:
        return {"recent_valid_acc_delta": None, "recent_valid_loss_delta": None}

    acc_first = sum(valid_acc[-10:-5]) / 5
    acc_last = sum(valid_acc[-5:]) / 5
    loss_first = sum(valid_loss[-10:-5]) / 5
    loss_last = sum(valid_loss[-5:]) / 5
    return {
        "recent_valid_acc_delta": acc_last - acc_first,
        "recent_valid_loss_delta": loss_last - loss_first,
    }


def summarize_by_model(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    best_by_model: dict[str, dict[str, object]] = {}
    for row in rows:
        model = row.get("MODEL", "")
        if model not in best_by_model or (
            row.get("valid_accuracy_last10_avg") or -1
        ) > (best_by_model[model].get("valid_accuracy_last10_avg") or -1):
            best_by_model[model] = row
    return best_by_model


def recommend(rows: list[dict[str, object]], target: float | None, logs_dir: Path) -> list[str]:
    top = best_rows(rows, limit=5)
    if not top:
        return ["No scored runs were found in the summary CSV."]

    best = top[0]
    model_best = summarize_by_model(rows)
    suggestions: list[str] = []
    best_acc = best["valid_accuracy_last10_avg"]
    trend = inspect_run_trend(logs_dir, best["time_stamp"])

    if target is not None and best_acc is not None and best_acc >= target:
        suggestions.append("Target is already met. Focus on reproducibility, best-checkpoint validation, and smaller ablations instead of large architecture changes.")
    else:
        if len(model_best) >= 2:
            ranked_models = sorted(
                model_best.values(),
                key=lambda row: row.get("valid_accuracy_last10_avg") or -1,
                reverse=True,
            )
            if len(ranked_models) >= 2:
                first = ranked_models[0]
                second = ranked_models[1]
                gap = (first.get("valid_accuracy_last10_avg") or 0) - (second.get("valid_accuracy_last10_avg") or 0)
                if gap > 0.003:
                    suggestions.append(
                        f"Prioritize `{first['MODEL']}` over `{second['MODEL']}`. Its best validation accuracy is higher by {gap:.4f}."
                    )

        if trend["recent_valid_acc_delta"] is not None and trend["recent_valid_acc_delta"] > 0.001:
            suggestions.append(
                f"Extend the current best configuration `{best['MODEL']}` at `lr={best['L_LATE']}`, `weight_decay={best.get('WEIGHT_DECAY')}`, `scheduler={best.get('SCHEDULER_NAME') or 'none'}` to a longer run because the last 10 epochs were still improving."
            )
        else:
            suggestions.append(
                f"Start from the best current configuration `{best['MODEL']}` with `epochs={best['NUM_EPOCHS']}`, `lr={best['L_LATE']}`, `weight_decay={best.get('WEIGHT_DECAY')}`, `scheduler={best.get('SCHEDULER_NAME') or 'none'}` and run a narrow tuning sweep around it."
            )

        if best.get("SCHEDULER_NAME"):
            suggestions.append(
                f"Keep `{best['SCHEDULER_NAME']}` in the next experiments. The current top run already uses it."
            )

        if target is not None and best_acc is not None:
            gap = target - best_acc
            if gap <= 0.01:
                suggestions.append("The remaining gap is under 1 percentage point. Prefer longer training, scheduler tuning, and augmentation ablations before trying a larger architectural jump.")
            else:
                suggestions.append("The remaining gap is still material. After one more tuning sweep, consider a stronger architecture or better regularization strategy if the target remains out of reach.")

    deduped = []
    seen = set()
    for suggestion in suggestions:
        if suggestion not in seen:
            deduped.append(suggestion)
            seen.add(suggestion)
    return deduped[:3]


def print_summary(rows: list[dict[str, object]], target: float | None, logs_dir: Path) -> None:
    top = best_rows(rows, limit=5)
    if not top:
        print("No comparable runs found.")
        return

    best = top[0]
    print("# FashionMNIST Target Summary")
    if target is None:
        print("- target_accuracy: not parsed from TARGET.md")
    else:
        gap = target - best["valid_accuracy_last10_avg"]
        print(f"- target_accuracy: {target:.4f}")
        print(f"- best_accuracy: {best['valid_accuracy_last10_avg']:.4f}")
        print(f"- gap_to_target: {gap:.4f}")

    print("\n## Best Run")
    print(f"- timestamp: {best['time_stamp']}")
    print(f"- model: {best['MODEL']}")
    print(f"- epochs: {best['NUM_EPOCHS']}")
    print(f"- learning_rate: {best['L_LATE']}")
    print(f"- weight_decay: {best.get('WEIGHT_DECAY')}")
    print(f"- scheduler: {best.get('SCHEDULER_NAME') or 'none'}")
    print(f"- valid_accuracy_last10_avg: {best['valid_accuracy_last10_avg']:.4f}")
    print(f"- valid_loss_last10_avg: {best.get('valid_loss_last10_avg'):.4f}")

    print("\n## Top Runs")
    for row in top:
        print(
            f"- {row['time_stamp']} | {row['MODEL']} | ep={row['NUM_EPOCHS']} | "
            f"lr={row['L_LATE']} | wd={row.get('WEIGHT_DECAY')} | "
            f"scheduler={row.get('SCHEDULER_NAME') or 'none'} | "
            f"val_acc={row['valid_accuracy_last10_avg']:.4f}"
        )

    print("\n## Next Experiments")
    for suggestion in recommend(rows, target, logs_dir):
        print(f"- {suggestion}")


def main():
    args = parse_args()
    target_path = Path(args.target)
    summary_csv = Path(args.summary_csv)
    logs_dir = Path(args.logs_dir)

    target = read_target_accuracy(target_path)
    rows = read_rows(summary_csv)
    print_summary(rows, target, logs_dir)


if __name__ == "__main__":
    main()
